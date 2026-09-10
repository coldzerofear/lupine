ARG CUDA_VERSION=13.3.1
ARG UBUNTU_VERSION=24.04
ARG CUDA_IMAGE_FLAVOR=devel
ARG ROCM_VERSION=7.2.4
ARG ROCM_SDK_IMAGE=rocm/dev-ubuntu-24.04:7.2.4@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054
ARG ROCM_SDK_PLATFORM=linux/amd64

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_IMAGE_FLAVOR}-ubuntu${UBUNTU_VERSION} AS cuda-sdk

FROM --platform=${ROCM_SDK_PLATFORM} ${ROCM_SDK_IMAGE} AS rocm-sdk

FROM ghcr.io/astral-sh/uv:0.9.18-python3.12-bookworm-slim@sha256:0b074d1ae15f5c3f1861354917d356e5afbd5a4c53c1190e81ad2f2add46e45b AS uv

FROM cuda-sdk AS cuda-ops

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /opt/lupine

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY ops/precompile.cmake ops/cuda_smemcpy.cu ops/smemcpy.h ops/smemcpy_dispatch.h /opt/lupine/ops/

RUN cmake \
      -DLUPINE_PRECOMPILED_OPS=/opt/lupine-precompiled-ops \
      -P /opt/lupine/ops/precompile.cmake

FROM ubuntu:${UBUNTU_VERSION} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_BUILD_TYPE=Release
ARG LUPINE_CLIENT_BUNDLE_INPUT
ARG CUDA_VERSION

# Device operations are precompiled in their own SDK stages. The main builder
# needs only API headers, link-time stubs, and the combined operation directory,
# so CUDA and ROCm compiler SDKs never have to coexist here.
COPY --from=cuda-sdk /usr/local/cuda/include/ /usr/local/cuda/include/
COPY --from=cuda-sdk /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so
COPY --from=cuda-ops /opt/lupine-precompiled-ops/ /opt/lupine-precompiled-ops/
COPY --from=rocm-sdk /opt/rocm/include/ /opt/rocm/include/
COPY --from=uv /usr/local/bin/uv /usr/local/bin/uv

ENV CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    libnghttp2-dev \
    libssl-dev \
    ninja-build \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lupine

COPY . /opt/lupine

RUN cmake -S /opt/lupine -B /opt/lupine/build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
      -DLUPINE_CUDA_DRIVER_LIBRARY="${CUDA_HOME}/lib64/stubs/libcuda.so" \
      -DLUPINE_CUDA_VERSION_OVERRIDE="${CUDA_VERSION}" \
      -DLUPINE_CLIENT_BUNDLE_INPUT="${LUPINE_CLIENT_BUNDLE_INPUT}" \
      -DLUPINE_PRECOMPILED_OPS=/opt/lupine-precompiled-ops

FROM builder AS client-build

RUN cmake --build /opt/lupine/build --parallel \
      --target lupine_cuda_client lupine_cudart_client lupine_nvml_client lupine_hip_client

FROM builder AS server-build

ARG LUPINE_CLIENT_BUNDLE_INPUT

RUN test -n "${LUPINE_CLIENT_BUNDLE_INPUT}"

RUN cmake --build /opt/lupine/build --parallel --target lupine_driver_server

FROM ubuntu:${UBUNTU_VERSION} AS client

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_VERSION
ARG NVIDIA_UTILS_PACKAGE=nvidia-utils-535
ARG NVIDIA_UTILS_VERSION=
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
    requested="$NVIDIA_UTILS_PACKAGE"; \
    if [ -n "$NVIDIA_UTILS_VERSION" ]; then \
      requested="${NVIDIA_UTILS_PACKAGE}=${NVIDIA_UTILS_VERSION}"; \
    fi; \
    if ! try_nvidia_utils "$requested"; then \
      for pkg in $(apt-cache search --names-only '^nvidia-utils-[0-9]+$' | awk '{print $1}' | sort -t- -k3 -rn); do \
        if try_nvidia_utils "$pkg"; then break; fi; \
      done; \
    fi; \
    cp /tmp/nvidia-utils/root/usr/bin/nvidia-smi /usr/bin/nvidia-smi; \
    rm -rf /var/lib/apt/lists/* /tmp/nvidia-utils

COPY --from=client-build /opt/lupine/build/libcuda.so.1 /opt/lupine/lib/libcuda.so.1
COPY --from=client-build /opt/lupine/build/libcudart.so* /opt/lupine/lib/
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

FROM ubuntu:${UBUNTU_VERSION} AS server

ARG DEBIAN_FRONTEND=noninteractive
ARG AMDGPU_INSTALL_VERSION=7.2.4.70204-1
ARG CUDA_KEYRING_VERSION=1.1-1
ARG CUDA_VERSION
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
    && apt-get install -y --no-install-recommends "cuda-compat-${cuda_series}" "cuda-cudart-${cuda_series}" \
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

ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/compat:/usr/local/cuda/lib64:/opt/rocm/lib
ENV LUPINE_PORT=14833
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
# Needed by the cudart gate below (FROM consumes the global ARG, RUN does not).
ARG CUDA_VERSION

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
      -DCMAKE_SHARED_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
      -DCMAKE_LIBRARY_PATH="${CUDA_HOME}/lib64/stubs"; \
    cmake --build /opt/lupine/build-static --parallel "$(nproc)" \
      --target lupine_cuda_client lupine_cudart_client lupine_nvml_client

RUN chmod +x /opt/lupine/deploy/check_static_client.sh \
    && /opt/lupine/deploy/check_static_client.sh \
         /opt/lupine/build-static/libcuda.so.1 \
         /opt/lupine/build-static/libnvidia-ml.so.1 \
         "${MAX_GLIBC}"

# libcudart.so.<major> is now a first-class codegen shim (lupine_cudart_client,
# built above), version-matched to this lane's CUDA (so.11.0 / so.12 / so.13)
# and linked against our libcuda.so.1 ($ORIGIN rpath). It ships alongside the
# other shims so an injected image without a CUDA runtime can use it. Same
# glibc-only gate as the other shims; no longer 13.x-only.
RUN set -eux; \
    so="$(ls /opt/lupine/build-static/libcudart.so.* 2>/dev/null \
            | grep -E 'libcudart\.so\.[0-9.]+$' | head -1)"; \
    test -n "$so"; \
    if readelf -d "$so" | grep NEEDED | \
         grep -vE 'lib(c|m|dl|rt|pthread)\.so|ld-linux|libcuda\.so\.1'; then \
      echo "libcudart has an unexpected dependency"; exit 1; \
    fi; \
    if nm -D --defined-only "$so" | awk '{print $3}' | \
         grep -vE '^(cuda|__cuda|libcudart\.so)'; then \
      echo "libcudart leaks non-cudart symbols"; exit 1; \
    fi; \
    ceiling="$(readelf -V "$so" | grep -oE 'GLIBC_[0-9.]+' | sed 's/GLIBC_//' | sort -V | tail -1)"; \
    if ! printf '%s\n%s\n' "$ceiling" "$MAX_GLIBC" | sort -C -V; then \
      echo "libcudart needs glibc $ceiling > $MAX_GLIBC"; exit 1; \
    fi

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
# Same extraction as the upstream nvidia-utils stage, but from a plain Ubuntu
# base: cuda-sdk is versioned by the build matrix and 11.8.0 has no
# ubuntu24.04 image, so referencing it here broke those lanes - and pulling a
# multi-GB devel image for one binary is waste anyway. nvidia-utils comes from
# Ubuntu's own "restricted" component, which the official image enables.
FROM ubuntu:${UBUNTU_VERSION} AS client-static-nvidia-utils

ARG DEBIAN_FRONTEND=noninteractive
ARG NVIDIA_UTILS_PACKAGE=nvidia-utils-535
ARG NVIDIA_UTILS_VERSION=

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

# nvidia-smi rides along with the shims: inside a remote pod its
# libnvidia-ml.so.1 resolves to the lupine shim, so queries go over RPC to the
# server. Taken from the same nvidia-utils stage the upstream client image
# uses. Checked here (the build stage has binutils): dependencies must stay
# glibc + libnvidia-ml, and its glibc floor must not exceed the shims' floor.
FROM client-static-build AS client-static-smi
COPY --from=client-static-nvidia-utils /nvidia-smi /opt/nvidia-smi
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
COPY --from=client-static-build /opt/lupine/build-static/libcudart.so.* /probe/

RUN /probe/loadprobe /probe/libcuda.so.1 /probe/libnvidia-ml.so.1 \
    && for c in /probe/libcudart.so.*; do [ -e "$c" ] && /probe/loadprobe "$c"; done \
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
COPY --from=client-static-build /opt/lupine/build-static/libcudart.so.* /artifacts/

RUN printf 'cuda_version=%s\nmin_glibc=%s\n' \
      "${CUDA_VERSION}" "${MAX_GLIBC}" > /artifacts/metadata \
    && ln -s libcuda.so.1 /artifacts/libcuda.so \
    && ln -s libnvidia-ml.so.1 /artifacts/libnvidia-ml.so \
    && cudart="$(ls /artifacts/libcudart.so.* | grep -E 'libcudart\.so\.[0-9.]+$' | head -1)" \
    && ln -s "$(basename "$cudart")" /artifacts/libcudart.so

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
# Directory (in the build context) with the per-platform native clients to
# embed in the server, laid out as <input>/lupine-client-<tag>/... exactly like
# the upstream server image. Empty => no bundles embedded (endpoint 503).
ARG LUPINE_CLIENT_BUNDLE_INPUT=

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"

# uv drives cmake/bundle_codegen.py, which turns each native client into the
# embedded-bundle source compiled into the server (upstream #691).
COPY --from=uv /usr/local/bin/uv /usr/local/bin/uv

# bundle_codegen.py (run by uv to embed the client bundles) is a pure-stdlib
# script needing only python >= 3.10. Install python3.12 and force uv to use the
# system interpreter, so the build never reaches out to GitHub for a managed
# CPython (that download 504'd in CI and is an unnecessary network dependency).
ENV UV_PYTHON_PREFERENCE=only-system

RUN dnf install -y --enablerepo=powertools \
        gcc gcc-c++ libstdc++-static make cmake binutils file tar gzip \
        python3.12 \
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
      -DLUPINE_CLIENT_BUNDLE_INPUT="${LUPINE_CLIENT_BUNDLE_INPUT}" \
      -DCMAKE_LIBRARY_PATH="${CUDA_HOME}/lib64/stubs"; \
    echo "==== build ===="; \
    cmake --build /opt/lupine/build-static-server --parallel "$(nproc)" \
      --target lupine_driver_server

RUN chmod +x /opt/lupine/deploy/check_static_server.sh \
    && /opt/lupine/deploy/check_static_server.sh \
         /opt/lupine/build-static-server/lupine_driver_server \
         "${MAX_GLIBC}"

# Run-probe AT THE GLIBC FLOOR on a bare rockylinux8-minimal (glibc 2.28), which
# is stricter than the final image's nvidia/cuda base: no libnghttp2, no
# libstdc++, no CUDA env. main() validates LUPINE_PORT and exits before touching
# the driver, so an invalid port is a complete load-and-run test -- a missing
# dependency exits 127 from the loader, a working binary exits 1 from the
# validation. The driver stub stands in for the injected libcuda.
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

# Runtime base = NVIDIA's CUDA rockylinux8 (same glibc 2.28 floor as
# rockylinux8-minimal, so the binary's ceiling still holds), NOT the minimal
# base. The minimal image lacks the CUDA driver environment the server needs at
# runtime -- the container runtime injects libcuda.so.1 into
# /usr/local/nvidia/lib64 and this base's LD_LIBRARY_PATH + cuda-compat make it
# resolvable, which the minimal base did not (server failed to find the driver).
# Bigger image, but required to run. Client bundles are embedded in the binary
# (LUPINE_CLIENT_BUNDLE_INPUT at build time, upstream #691); served over
# HTTP/1.x on the RPC port at /.well-known/lupine/client/v1/<platform>.
FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_IMAGE_FLAVOR}-rockylinux8 AS server-static

ARG CUDA_VERSION
ARG MAX_GLIBC=2.28

LABEL org.opencontainers.image.title="lupine-server-static"
LABEL org.opencontainers.image.description="LUPINE server (glibc-only binary) on the CUDA rockylinux8 driver base"
LABEL org.opencontainers.image.source="https://github.com/lupinemachines/lupine"
LABEL org.opencontainers.image.version="${CUDA_VERSION}-static"
LABEL io.lupine.cuda-version="${CUDA_VERSION}"
LABEL io.lupine.min-glibc="${MAX_GLIBC}"

# The run-probe produces no artifact we ship; copying its marker makes it a hard
# build dependency so the probe cannot be skipped by stage pruning.
COPY --from=server-static-runprobe /probe/runprobe-passed /opt/lupine/.runprobe-passed
COPY --from=server-static-build /opt/lupine/build-static-server/lupine_driver_server /opt/lupine/bin/lupine_driver_server

RUN chmod +x /opt/lupine/bin/lupine_driver_server

ENV LUPINE_PORT=14833
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 14833

ENTRYPOINT ["/opt/lupine/bin/lupine_driver_server"]
