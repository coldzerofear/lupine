ARG CUDA_VERSION=13.3.1
ARG UBUNTU_VERSION=24.04
ARG CUDA_IMAGE_FLAVOR=devel
ARG ROCM_VERSION=7.2.4
ARG ROCM_SDK_IMAGE=rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054
ARG ROCM_SDK_PLATFORM=linux/amd64

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_IMAGE_FLAVOR}-ubuntu${UBUNTU_VERSION} AS cuda-sdk

FROM --platform=${ROCM_SDK_PLATFORM} ${ROCM_SDK_IMAGE} AS rocm-sdk

FROM cuda-sdk AS cuda-ops

WORKDIR /opt/lupine

COPY ops/smemcpy.cu ops/smemcpy.h ops/smemcpy_dispatch.h /opt/lupine/ops/

# Precompile CUDA operations before entering the SDK-neutral builder. Include
# native code for every architecture accepted by this toolkit and PTX for the
# oldest one as a forward-compatible fallback.
RUN set -eux; \
    set --; \
    oldest=""; \
    for code in $(nvcc --list-gpu-code); do \
      arch="${code#sm_}"; \
      if [ -z "$oldest" ] || [ "$arch" -lt "$oldest" ]; then \
        oldest="$arch"; \
      fi; \
      set -- "$@" \
        "--generate-code=arch=compute_${arch},code=sm_${arch}"; \
    done; \
    test -n "$oldest"; \
    set -- "$@" \
      "--generate-code=arch=compute_${oldest},code=compute_${oldest}"; \
    mkdir -p /opt/lupine-precompiled-ops/cuda; \
    nvcc -std=c++17 --fatbin "$@" \
      -I/opt/lupine /opt/lupine/ops/smemcpy.cu \
      -o /tmp/lupine_smemcpy.fatbin; \
    bin2c --const --name lupine_smemcpy_fatbin \
      /tmp/lupine_smemcpy.fatbin \
      > /opt/lupine-precompiled-ops/cuda/smemcpy.cpp; \
    printf '\nextern "C" const void *lupine_cuda_smemcpy_image() {\n  return lupine_smemcpy_fatbin;\n}\n' \
      >> /opt/lupine-precompiled-ops/cuda/smemcpy.cpp; \
    rm /tmp/lupine_smemcpy.fatbin

FROM ubuntu:${UBUNTU_VERSION} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_BUILD_TYPE=Release
ARG CUDA_VERSION

# Device operations are precompiled in their own SDK stages. The main builder
# needs only API headers, link-time stubs, and the combined operation directory,
# so CUDA and ROCm compiler SDKs never have to coexist here.
COPY --from=cuda-sdk /usr/local/cuda/include/ /usr/local/cuda/include/
COPY --from=cuda-sdk /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so
COPY --from=cuda-ops /opt/lupine-precompiled-ops/ /opt/lupine-precompiled-ops/
COPY --from=rocm-sdk /opt/rocm/include/ /opt/rocm/include/

ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    libnghttp2-dev \
    libssl-dev \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lupine

COPY . /opt/lupine

RUN cmake -S /opt/lupine -B /opt/lupine/build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
      -DLUPINE_CUDA_DRIVER_LIBRARY="${CUDA_HOME}/lib64/stubs/libcuda.so" \
      -DLUPINE_CUDA_VERSION_OVERRIDE="${CUDA_VERSION}" \
      -DLUPINE_PRECOMPILED_OPS=/opt/lupine-precompiled-ops

FROM builder AS client-build

RUN cmake --build /opt/lupine/build --parallel \
      --target lupine_cuda_client lupine_nvml_client lupine_hip_client

RUN test -e /opt/lupine/build/libcuda.so.1 \
    && test -e /opt/lupine/build/libnvidia-ml.so.1 \
    && test -e /opt/lupine/build/libamdhip64.so.1 \
    && ln -sf libcuda.so.1 /opt/lupine/build/libcuda.so \
    && ln -sf libnvidia-ml.so.1 /opt/lupine/build/libnvidia-ml.so \
    && ln -sf libamdhip64.so.1 /opt/lupine/build/libamdhip64.so \
    && ! nm -D --defined-only /opt/lupine/build/libcuda.so.1 \
      | awk '{print $3}' \
      | grep -E '^cuda'

FROM builder AS server-build

RUN cmake --build /opt/lupine/build --parallel --target lupine_driver_server

RUN test -x /opt/lupine/build/lupine_driver_server

FROM cuda-sdk AS nvidia-utils

ARG DEBIAN_FRONTEND=noninteractive
ARG NVIDIA_UTILS_PACKAGE=nvidia-utils-535
ARG NVIDIA_UTILS_VERSION=

# Ubuntu periodically turns an older nvidia-utils-NNN into an empty
# transitional package (Depends on a newer NNN, no binaries of its own) as
# driver branches age out, so the pinned NVIDIA_UTILS_PACKAGE can silently
# stop shipping nvidia-smi. Try the pin first, then fall back to whichever
# nvidia-utils-NNN (newest first) actually contains it.
RUN set -eux; \
    apt-get update; \
    mkdir -p /tmp/nvidia-utils; \
    cd /tmp/nvidia-utils; \
    try_nvidia_utils() { \
      rm -f ./*.deb; \
      rm -rf /tmp/nvidia-utils/root; \
      apt-get download "$1" >/dev/null 2>&1 || return 1; \
      dpkg-deb -x ./*.deb /tmp/nvidia-utils/root || return 1; \
      test -x /tmp/nvidia-utils/root/usr/bin/nvidia-smi; \
    }; \
    found=""; \
    if [ -n "$NVIDIA_UTILS_VERSION" ]; then \
      try_nvidia_utils "${NVIDIA_UTILS_PACKAGE}=${NVIDIA_UTILS_VERSION}" && found=1; \
    else \
      try_nvidia_utils "${NVIDIA_UTILS_PACKAGE}" && found=1; \
    fi; \
    if [ -z "$found" ]; then \
      for pkg in $(apt-cache search --names-only '^nvidia-utils-[0-9]+$' | awk '{print $1}' | sort -t- -k3 -rn); do \
        if try_nvidia_utils "$pkg"; then found=1; break; fi; \
      done; \
    fi; \
    test -n "$found"; \
    cp /tmp/nvidia-utils/root/usr/bin/nvidia-smi /nvidia-smi; \
    chmod +x /nvidia-smi; \
    rm -rf /var/lib/apt/lists/* /tmp/nvidia-utils

FROM ubuntu:${UBUNTU_VERSION} AS client

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_VERSION
ARG ROCM_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.title="lupine-client"
LABEL org.opencontainers.image.description="LUPINE client runtime with CUDA, NVML, and HIP shims"
LABEL org.opencontainers.image.source="https://github.com/lupinemachines/lupine"
LABEL org.opencontainers.image.version="${CUDA_VERSION}-rocm-${ROCM_VERSION}-ubuntu${UBUNTU_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    libgcc-s1 \
    libnghttp2-14 \
    libstdc++6 \
    # libssl3 on jammy, libssl3t64 on noble.
    && (apt-get install -y --no-install-recommends libssl3 || apt-get install -y --no-install-recommends libssl3t64) \
    && rm -rf /var/lib/apt/lists/*

COPY --from=nvidia-utils /nvidia-smi /usr/bin/nvidia-smi

COPY --from=client-build /opt/lupine/build/libcuda.so.1 /opt/lupine/lib/libcuda.so.1
COPY --from=client-build /opt/lupine/build/libnvidia-ml.so.1 /opt/lupine/lib/libnvidia-ml.so.1
COPY --from=client-build /opt/lupine/build/libamdhip64.so.1 /opt/lupine/lib/libamdhip64.so.1

RUN ln -sf /opt/lupine/lib/libcuda.so.1 /opt/lupine/lib/libcuda.so \
    && ln -sf /opt/lupine/lib/libnvidia-ml.so.1 /opt/lupine/lib/libnvidia-ml.so \
    && ln -sf /opt/lupine/lib/libamdhip64.so.1 /opt/lupine/lib/libamdhip64.so

ENV LUPINE_LIBCUDA=/opt/lupine/lib/libcuda.so.1
ENV LUPINE_LIB=/opt/lupine/lib/libcuda.so.1
ENV LUPINE_LIBHIP=/opt/lupine/lib/libamdhip64.so.1
ENV LD_LIBRARY_PATH=/opt/lupine/lib

ENTRYPOINT []
CMD ["bash"]

FROM client AS client-slim

ARG CUDA_VERSION
ARG ROCM_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.description="LUPINE SDK-free client runtime with CUDA, NVML, and HIP shims"
LABEL org.opencontainers.image.version="${CUDA_VERSION}-rocm-${ROCM_VERSION}-ubuntu${UBUNTU_VERSION}-slim"

FROM ubuntu:${UBUNTU_VERSION} AS server

ARG DEBIAN_FRONTEND=noninteractive
ARG AMDGPU_INSTALL_VERSION=7.2.4.70204-1
ARG CUDA_KEYRING_VERSION=1.1-1
ARG CUDA_VERSION
ARG LUPINE_REQUIRE_CLIENT_BUNDLES=0
ARG ROCM_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.title="lupine-server"
LABEL org.opencontainers.image.description="LUPINE CUDA and ROCm server runtime"
LABEL org.opencontainers.image.source="https://github.com/lupinemachines/lupine"
LABEL org.opencontainers.image.version="${CUDA_VERSION}-rocm-${ROCM_VERSION}-ubuntu${UBUNTU_VERSION}"

# NVIDIA's container runtime supplies the host driver ahead of the compatibility
# library in LD_LIBRARY_PATH. The compatibility package also lets the unified
# binary start on AMD-only hosts, where no NVIDIA driver is mounted.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    libgcc-s1 \
    libnghttp2-14 \
    libstdc++6 \
    wget \
    # libssl3 on jammy, libssl3t64 on noble.
    && (apt-get install -y --no-install-recommends libssl3 || apt-get install -y --no-install-recommends libssl3t64) \
    && arch="$(dpkg --print-architecture)" \
    && case "$arch" in \
         amd64) cuda_repo_arch=x86_64 ;; \
         arm64) cuda_repo_arch=sbsa ;; \
         *) echo "Unsupported CUDA architecture: $arch" >&2; exit 1 ;; \
       esac \
    && ubuntu_repo="ubuntu$(printf '%s' "${UBUNTU_VERSION}" | tr -d '.')" \
    && cuda_series="$(printf '%s' "${CUDA_VERSION}" | awk -F. '{print $1 "-" $2}')" \
    && wget -q \
         "https://developer.download.nvidia.com/compute/cuda/repos/${ubuntu_repo}/${cuda_repo_arch}/cuda-keyring_${CUDA_KEYRING_VERSION}_all.deb" \
         -O /tmp/cuda-keyring.deb \
    && apt-get install -y --no-install-recommends /tmp/cuda-keyring.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends "cuda-compat-${cuda_series}" \
    && cuda_series_dot="$(printf '%s' "${CUDA_VERSION}" | awk -F. '{print $1 "." $2}')" \
    && ln -sfn "cuda-${cuda_series_dot}" /usr/local/cuda \
    && if [ "$arch" = amd64 ]; then \
         . /etc/os-release; \
         wget -q \
           "https://repo.radeon.com/amdgpu-install/${ROCM_VERSION}/ubuntu/${VERSION_CODENAME}/amdgpu-install_${AMDGPU_INSTALL_VERSION}_all.deb" \
           -O /tmp/amdgpu-install.deb; \
         apt-get install -y --no-install-recommends /tmp/amdgpu-install.deb; \
         apt-get update; \
         apt-get install -y --no-install-recommends hip-runtime-amd; \
         apt-get purge -y amdgpu-install; \
       fi \
    && apt-get purge -y wget \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/*.deb

COPY --from=server-build /opt/lupine/build/lupine_driver_server /opt/lupine/bin/lupine_driver_server
COPY client-bundles/ /opt/lupine/client-bundles/

RUN set -eux; \
    if [ "${LUPINE_REQUIRE_CLIENT_BUNDLES}" = 1 ]; then \
      for platform in \
        linux/amd64 linux/arm64 \
        macos/amd64 macos/arm64 \
        windows/amd64 windows/arm64; do \
        test -s "/opt/lupine/client-bundles/${platform}/client.zip"; \
        test -s "/opt/lupine/client-bundles/${platform}/client.zip.etag"; \
        test -s "/opt/lupine/client-bundles/${platform}/client.zip.digest"; \
      done; \
    fi

RUN chmod +x /opt/lupine/bin/lupine_driver_server

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/compat:/opt/rocm/lib
ENV LUPINE_PORT=14833
ENV LUPINE_CLIENT_BUNDLE_DIR=/opt/lupine/client-bundles
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 14833

ENTRYPOINT ["/opt/lupine/bin/lupine_driver_server"]


# ---------------------------------------------------------------------------
# Self-contained ("static") client shims.
#
# The regular client images above carry their own runtime (libnghttp2, libssl,
# libstdc++). That works when the app runs INSIDE those images -- but the shims
# are also injected into arbitrary user images (k8s device injection), whose
# library sets we do not control. There, dynamic deps either fail to resolve or
# have to be shipped alongside, where they shadow the image's own copies for
# every process in the container. These stages produce shims whose only runtime
# dependency is glibc: nghttp2 / OpenSSL / libstdc++ / libgcc are linked in
# statically (LUPINE_STATIC_DEPS=ON) and hidden from .dynsym by the existing
# export version scripts.
#
# Built on rockylinux8 rather than Ubuntu, deliberately: its glibc 2.28 is the
# lowest floor NVIDIA publishes devel images for across the whole CUDA matrix
# (11.7-13.1, amd64+arm64), and an artifact only loads on glibc >= its
# builder's. gcc-toolset supplies a newer compiler where nvcc requires one;
# its libstdc++ delta links statically by design, so the floor stays 2.28.
# ---------------------------------------------------------------------------

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_IMAGE_FLAVOR}-rockylinux8 AS client-static-build

ARG CMAKE_BUILD_TYPE=Release
ARG NGHTTP2_VERSION=1.64.0
ARG NGHTTP2_SHA256=20e73f3cf9db3f05988996ac8b3a99ed529f4565ca91a49eb0550498e10621e8
ARG OPENSSL_VERSION=3.0.18
ARG OPENSSL_SHA256=d80c34f5cf902dccf1f1b5df5ebb86d0392e37049e5d73df1b3abae72e4ffe8b
# Declared glibc ceiling; check_static_client.sh fails the build if the linked
# result references anything newer (e.g. someone swaps in a newer base).
ARG MAX_GLIBC=2.28
# Set to a gcc-toolset package name (e.g. gcc-toolset-13) when this CUDA
# version's nvcc rejects the system gcc 8.5. Empty = system gcc.
ARG GCC_TOOLSET=

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"

# libstdc++-static lives in PowerTools (disabled by default on Rocky 8), unlike
# Ubuntu where libstdc++-dev ships the .a. Without it, -static-libstdc++ makes
# ld hunt for libstdc++.a, fail with "cannot find -lstdc++", and the whole
# matrix dies at link time. gcc-toolset ships its own libstdc++.a via its
# -libstdc++-devel (pulled by -gcc-c++), so the toolset lane needs nothing
# extra.
RUN dnf install -y --enablerepo=powertools \
        gcc gcc-c++ libstdc++-static make cmake perl binutils file tar gzip \
    && if [ -n "${GCC_TOOLSET}" ]; then \
         dnf install -y "${GCC_TOOLSET}-gcc" "${GCC_TOOLSET}-gcc-c++"; \
       fi \
    && dnf clean all

# nghttp2: C library only, static, PIC (it ends up inside a shared object).
RUN set -eux; \
    curl -fsSL -o nghttp2.tar.gz \
      "https://github.com/nghttp2/nghttp2/releases/download/v${NGHTTP2_VERSION}/nghttp2-${NGHTTP2_VERSION}.tar.gz"; \
    echo "${NGHTTP2_SHA256}  nghttp2.tar.gz" | sha256sum -c -; \
    tar xzf nghttp2.tar.gz; \
    cd "nghttp2-${NGHTTP2_VERSION}"; \
    ./configure --prefix=/opt/static-deps --enable-lib-only \
                --enable-static --disable-shared --with-pic; \
    make -j"$(nproc)"; \
    make install; \
    cd ..; rm -rf "nghttp2-${NGHTTP2_VERSION}" nghttp2.tar.gz

# OpenSSL: static, PIC, no loadable modules (nothing to dlopen at runtime).
# install_sw skips man pages. `./config` autodetects amd64 vs arm64.
RUN set -eux; \
    curl -fsSL -o openssl.tar.gz \
      "https://github.com/openssl/openssl/releases/download/openssl-${OPENSSL_VERSION}/openssl-${OPENSSL_VERSION}.tar.gz"; \
    echo "${OPENSSL_SHA256}  openssl.tar.gz" | sha256sum -c -; \
    tar xzf openssl.tar.gz; \
    cd "openssl-${OPENSSL_VERSION}"; \
    ./config --prefix=/opt/static-deps --libdir=lib \
             no-shared no-module no-tests -fPIC; \
    make -j"$(nproc)" build_sw; \
    make install_sw; \
    cd ..; rm -rf "openssl-${OPENSSL_VERSION}" openssl.tar.gz

WORKDIR /opt/lupine
COPY . /opt/lupine

# The toolset gcc goes on PATH by hand rather than via `scl_source enable`:
# scl_source trips `set -u` ("_recursion: unbound variable") and takes the whole
# RUN down with exit 1 before cmake even starts. Prepending its bin dir is all
# scl_source does for our purposes; cmake and nvcc pick the compiler from PATH.
# The configure/build markers exist because BuildKit only shows the failing
# step tail -- "exit code 2" alone told us nothing the first time this broke.
RUN set -eux; \
    if [ -n "${GCC_TOOLSET}" ]; then export PATH="/opt/rh/${GCC_TOOLSET}/root/usr/bin:${PATH}"; fi; \
    cmake -S /opt/lupine -B /opt/lupine/build-static \
      -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
      -DLUPINE_STATIC_DEPS=ON \
      -DNGHTTP2_INCLUDE_DIR=/opt/static-deps/include \
      -DNGHTTP2_LIBRARY=/opt/static-deps/lib/libnghttp2.a \
      -DOPENSSL_ROOT_DIR=/opt/static-deps \
      -DCMAKE_LIBRARY_PATH="${CUDA_HOME}/lib64/stubs"; \
    cmake --build /opt/lupine/build-static --parallel "$(nproc)" \
      --target lupine_cuda_client lupine_nvml_client

RUN chmod +x /opt/lupine/deploy/check_static_client.sh \
    && /opt/lupine/deploy/check_static_client.sh \
         /opt/lupine/build-static/libcuda.so.1 \
         /opt/lupine/build-static/libnvidia-ml.so.1 \
         "${MAX_GLIBC}"

# Load-probe helper: RTLD_NOW forces every relocation, so a missing dependency
# or undefined symbol fails at build time, not in a user pod.
RUN printf '%s\n' \
      '#include <dlfcn.h>' \
      '#include <stdio.h>' \
      'int main(int argc, char **argv) {' \
      '  for (int i = 1; i < argc; i++) {' \
      '    if (!dlopen(argv[i], RTLD_NOW | RTLD_LOCAL)) {' \
      '      fprintf(stderr, "FAIL %s: %s\n", argv[i], dlerror());' \
      '      return 1;' \
      '    }' \
      '    printf("ok %s\n", argv[i]);' \
      '  }' \
      '  return 0;' \
      '}' > /tmp/loadprobe.c \
    && gcc -o /opt/lupine/build-static/loadprobe /tmp/loadprobe.c -ldl

# Bare image AT THE GLIBC FLOOR (rocky8-minimal = 2.28), deliberately: no
# libnghttp2, no libssl, no libstdc++ guarantees beyond the base, no CUDA. If
# the shims load here under RTLD_NOW, they load anywhere with glibc >= 2.28.
# nvidia-smi rides along with the shims: inside a remote pod its
# libnvidia-ml.so.1 resolves to the lupine shim, so queries go over RPC to the
# server. Taken from the same nvidia-utils stage the upstream client image
# uses. Checked here (the build stage has binutils): dependencies must stay
# glibc + libnvidia-ml, and its glibc floor must not exceed the shims' floor.
FROM client-static-build AS client-static-smi
COPY --from=nvidia-utils /nvidia-smi /opt/nvidia-smi
RUN set -eux; \
    if readelf -d /opt/nvidia-smi | grep NEEDED | \
         grep -vE 'lib(c|m|dl|rt|pthread)\.so|ld-linux|libnvidia-ml\.so\.1'; then \
      echo "nvidia-smi has an unexpected dependency"; exit 1; \
    fi; \
    ceiling="$(readelf -V /opt/nvidia-smi | grep -oE 'GLIBC_[0-9.]+' | sed 's/GLIBC_//' | sort -V | tail -1)"; \
    if ! printf '%s\n%s\n' "$ceiling" "$MAX_GLIBC" | sort -C -V; then \
      echo "nvidia-smi needs glibc $ceiling > $MAX_GLIBC"; exit 1; \
    fi

FROM rockylinux:8-minimal AS client-static-loadtest

COPY --from=client-static-build /opt/lupine/build-static/libcuda.so.1 /probe/libcuda.so.1
COPY --from=client-static-build /opt/lupine/build-static/libnvidia-ml.so.1 /probe/libnvidia-ml.so.1
COPY --from=client-static-build /opt/lupine/build-static/loadprobe /probe/loadprobe

RUN /probe/loadprobe /probe/libcuda.so.1 /probe/libnvidia-ml.so.1 \
    && touch /probe/loadtest-passed

# Final artifact carrier. busybox so an init container can `cp -a` the
# artifacts into a shared volume; nothing here ever executes the shims.
FROM busybox:stable-glibc AS client-static

ARG CUDA_VERSION
ARG MAX_GLIBC=2.28

LABEL org.opencontainers.image.title="lupine-client-static"
LABEL org.opencontainers.image.description="Self-contained LUPINE client shims (glibc-only runtime deps)"
LABEL org.opencontainers.image.source="https://github.com/lupinemachines/lupine"
LABEL org.opencontainers.image.version="${CUDA_VERSION}-static"
LABEL io.lupine.cuda-version="${CUDA_VERSION}"
LABEL io.lupine.min-glibc="${MAX_GLIBC}"

# The loadtest stage produces no artifact we ship; copying its marker makes it
# a hard build dependency so the probe cannot be skipped by stage pruning.
COPY --from=client-static-loadtest /probe/loadtest-passed /artifacts/.loadtest-passed
COPY --from=client-static-build /opt/lupine/build-static/libcuda.so.1 /artifacts/libcuda.so.1
COPY --from=client-static-build /opt/lupine/build-static/libnvidia-ml.so.1 /artifacts/libnvidia-ml.so.1
COPY --from=client-static-smi /opt/nvidia-smi /artifacts/nvidia-smi

RUN printf 'cuda_version=%s\nmin_glibc=%s\n' \
      "${CUDA_VERSION}" "${MAX_GLIBC}" > /artifacts/metadata \
    && ln -s libcuda.so.1 /artifacts/libcuda.so \
    && ln -s libnvidia-ml.so.1 /artifacts/libnvidia-ml.so

CMD ["sh", "-c", "cp -a /artifacts/. \"${ARTIFACTS_DEST:-/target}/\" && echo copied to ${ARTIFACTS_DEST:-/target}"]


# ---------------------------------------------------------------------------
# Self-contained ("static") server.
#
# The server stage above inherits nvidia/cuda:*-runtime purely to satisfy the
# loader, which is several GB for a binary that needs almost none of it: the
# server calls the driver API, and libcuda/libnvidia-ml come from the host via
# nvidia-container-runtime, not from the image. Link nghttp2 and the C++
# runtime in (LUPINE_STATIC_DEPS=ON, same flag the client shims use) and the
# only thing left to satisfy is glibc -- which a minimal base already has.
#
# Built on rockylinux8 for the same reason as the client shims: glibc 2.28 is
# the lowest floor NVIDIA publishes devel images for across the CUDA matrix,
# and a binary runs only on glibc >= its builder's. gcc-toolset supplies a
# newer compiler where nvcc requires one; -static-libstdc++ keeps its newer
# libstdc++ out of the runtime requirements, so the floor stays 2.28.
#
# OpenSSL is not built here, unlike the client lane: the server is plaintext by
# design (front it with a TLS proxy), so nghttp2 is the only dependency to
# build from source.
# ---------------------------------------------------------------------------

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_IMAGE_FLAVOR}-rockylinux8 AS server-static-build

ARG CMAKE_BUILD_TYPE=Release
ARG NGHTTP2_VERSION=1.64.0
ARG NGHTTP2_SHA256=20e73f3cf9db3f05988996ac8b3a99ed529f4565ca91a49eb0550498e10621e8
# Declared glibc ceiling; check_static_server.sh fails the build if the linked
# result references anything newer (e.g. someone swaps in a newer base).
ARG MAX_GLIBC=2.28
# Set to a gcc-toolset package name (e.g. gcc-toolset-13) when this CUDA
# version's nvcc rejects the system gcc 8.5. Empty = system gcc.
ARG GCC_TOOLSET=

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"

RUN dnf install -y --enablerepo=powertools \
        gcc gcc-c++ libstdc++-static make cmake binutils file tar gzip \
    && if [ -n "${GCC_TOOLSET}" ]; then \
         dnf install -y "${GCC_TOOLSET}-gcc" "${GCC_TOOLSET}-gcc-c++"; \
       fi \
    && dnf clean all

# nghttp2: C library only, static, PIC (it is linked into a position-independent
# executable).
RUN set -eux; \
    curl -fsSL -o nghttp2.tar.gz \
      "https://github.com/nghttp2/nghttp2/releases/download/v${NGHTTP2_VERSION}/nghttp2-${NGHTTP2_VERSION}.tar.gz"; \
    echo "${NGHTTP2_SHA256}  nghttp2.tar.gz" | sha256sum -c -; \
    tar xzf nghttp2.tar.gz; \
    cd "nghttp2-${NGHTTP2_VERSION}"; \
    ./configure --prefix=/opt/static-deps --enable-lib-only \
                --enable-static --disable-shared --with-pic; \
    make -j"$(nproc)"; \
    make install; \
    cd ..; rm -rf "nghttp2-${NGHTTP2_VERSION}" nghttp2.tar.gz

WORKDIR /opt/lupine
COPY . /opt/lupine

# CMAKE_LIBRARY_PATH points at the driver stubs so CUDA::cuda_driver resolves at
# link time; the real libcuda.so.1 is injected by the container runtime. See the
# client-static stage for why gcc-toolset goes on PATH by hand.
RUN set -eux; \
    if [ -n "${GCC_TOOLSET}" ]; then export PATH="/opt/rh/${GCC_TOOLSET}/root/usr/bin:${PATH}"; fi; \
    echo "==== configure ($(gcc --version | head -1)) ===="; \
    cmake -S /opt/lupine -B /opt/lupine/build-static-server \
      -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
      -DLUPINE_STATIC_DEPS=ON \
      -DNGHTTP2_INCLUDE_DIR=/opt/static-deps/include \
      -DNGHTTP2_LIBRARY=/opt/static-deps/lib/libnghttp2.a \
      -DCMAKE_LIBRARY_PATH="${CUDA_HOME}/lib64/stubs"; \
    echo "==== build ===="; \
    cmake --build /opt/lupine/build-static-server --parallel "$(nproc)" \
      --target lupine_driver_server

RUN chmod +x /opt/lupine/deploy/check_static_server.sh \
    && /opt/lupine/deploy/check_static_server.sh \
         /opt/lupine/build-static-server/lupine_driver_server \
         "${MAX_GLIBC}"

# Run-probe AT THE GLIBC FLOOR, on the same bare base the final image uses: no
# libnghttp2, no libstdc++, no CUDA. main() validates LUPINE_PORT and exits
# before it touches the driver, so an invalid port is a complete load-and-run
# test -- a missing dependency exits 127 from the loader, a working binary exits
# 1 from the validation. The driver stub stands in for the injected libcuda.
FROM rockylinux:8-minimal AS server-static-runprobe

COPY --from=server-static-build /opt/lupine/build-static-server/lupine_driver_server /probe/lupine_driver_server
COPY --from=server-static-build /usr/local/cuda/lib64/stubs/libcuda.so /probe/stubs/libcuda.so.1

RUN LD_LIBRARY_PATH=/probe/stubs LUPINE_PORT=not-a-port /probe/lupine_driver_server; \
    rc=$?; \
    if [ "$rc" -ne 1 ]; then \
      echo "FAIL: expected exit 1 from LUPINE_PORT validation, got $rc"; \
      exit 1; \
    fi; \
    touch /probe/runprobe-passed

FROM rockylinux:8-minimal AS server-static

ARG CUDA_VERSION
ARG MAX_GLIBC=2.28
# Mirrors the upstream server image (#689): 1 makes the build fail unless every
# platform bundle is present, which the publish workflow guarantees by
# downloading the python.yml artifact first. Dev builds leave it 0 and the
# server answers the client endpoint with 503.
ARG LUPINE_REQUIRE_CLIENT_BUNDLES=0

LABEL org.opencontainers.image.title="lupine-server-static"
LABEL org.opencontainers.image.description="Self-contained LUPINE server (glibc-only runtime deps)"
LABEL org.opencontainers.image.source="https://github.com/lupinemachines/lupine"
LABEL org.opencontainers.image.version="${CUDA_VERSION}-static"
LABEL io.lupine.cuda-version="${CUDA_VERSION}"
LABEL io.lupine.min-glibc="${MAX_GLIBC}"

# The run-probe produces no artifact we ship; copying its marker makes it a hard
# build dependency so the probe cannot be skipped by stage pruning.
COPY --from=server-static-runprobe /probe/runprobe-passed /opt/lupine/.runprobe-passed
COPY --from=server-static-build /opt/lupine/build-static-server/lupine_driver_server /opt/lupine/bin/lupine_driver_server
# Server-selected native client bundles (#689), served over HTTP/1.x on the RPC
# port at /.well-known/lupine/client/v1/<platform>. Same layout and gate as the
# upstream server stage so the static image is a drop-in for that endpoint.
COPY client-bundles/ /opt/lupine/client-bundles/

RUN set -eux; \
    if [ "${LUPINE_REQUIRE_CLIENT_BUNDLES}" = 1 ]; then \
      for platform in \
        linux/amd64 linux/arm64 \
        macos/amd64 macos/arm64 \
        windows/amd64 windows/arm64; do \
        test -s "/opt/lupine/client-bundles/${platform}/client.zip"; \
        test -s "/opt/lupine/client-bundles/${platform}/client.zip.etag"; \
        test -s "/opt/lupine/client-bundles/${platform}/client.zip.digest"; \
      done; \
    fi

RUN chmod +x /opt/lupine/bin/lupine_driver_server

ENV LUPINE_PORT=14833
ENV LUPINE_CLIENT_BUNDLE_DIR=/opt/lupine/client-bundles
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 14833

ENTRYPOINT ["/opt/lupine/bin/lupine_driver_server"]