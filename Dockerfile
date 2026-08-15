ARG CUDA_VERSION=13.1.0
ARG UBUNTU_VERSION=24.04
ARG CUDA_IMAGE_FLAVOR=devel
ARG CUDA_RUNTIME_IMAGE_FLAVOR=runtime

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_IMAGE_FLAVOR}-ubuntu${UBUNTU_VERSION} AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_BUILD_TYPE=Release

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}"

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
      -DCMAKE_LIBRARY_PATH="${CUDA_HOME}/lib64/stubs"

FROM builder AS client-build

RUN cmake --build /opt/lupine/build --parallel --target lupine_driver lupine_nvml

RUN test -e /opt/lupine/build/libcuda.so.1 \
    && test -e /opt/lupine/build/libnvidia-ml.so.1 \
    && ln -sf libcuda.so.1 /opt/lupine/build/libcuda.so \
    && ln -sf libnvidia-ml.so.1 /opt/lupine/build/libnvidia-ml.so \
    && ! nm -D --defined-only /opt/lupine/build/libcuda.so.1 \
      | awk '{print $3}' \
      | grep -E '^cuda'

FROM builder AS server-build

RUN cmake --build /opt/lupine/build --parallel --target lupine_driver_server

RUN test -x /opt/lupine/build/lupine_driver_server

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_RUNTIME_IMAGE_FLAVOR}-ubuntu${UBUNTU_VERSION} AS nvidia-utils

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

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_RUNTIME_IMAGE_FLAVOR}-ubuntu${UBUNTU_VERSION} AS client

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.title="lupine-client"
LABEL org.opencontainers.image.description="LUPINE client runtime with driver-only libcuda shim"
LABEL org.opencontainers.image.source="https://github.com/lupinemachines/lupine"
LABEL org.opencontainers.image.version="${CUDA_VERSION}-ubuntu${UBUNTU_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    libnghttp2-14 \
    # libssl3 on jammy, libssl3t64 on noble.
    && (apt-get install -y --no-install-recommends libssl3 || apt-get install -y --no-install-recommends libssl3t64) \
    && rm -rf /var/lib/apt/lists/*

COPY --from=nvidia-utils /nvidia-smi /usr/bin/nvidia-smi

COPY --from=client-build /opt/lupine/build/libcuda.so.1 /opt/lupine/lib/libcuda.so.1
COPY --from=client-build /opt/lupine/build/libnvidia-ml.so.1 /opt/lupine/lib/libnvidia-ml.so.1

RUN ln -sf /opt/lupine/lib/libcuda.so.1 /opt/lupine/lib/libcuda.so \
    && ln -sf /opt/lupine/lib/libnvidia-ml.so.1 /opt/lupine/lib/libnvidia-ml.so

ENV LUPINE_LIBCUDA=/opt/lupine/lib/libcuda.so.1
ENV LUPINE_LIB=/opt/lupine/lib/libcuda.so.1
ENV LD_LIBRARY_PATH=/opt/lupine/lib:${LD_LIBRARY_PATH}

ENTRYPOINT []
CMD ["bash"]

FROM ubuntu:${UBUNTU_VERSION} AS client-slim

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.title="lupine-client"
LABEL org.opencontainers.image.description="LUPINE slim client runtime with driver-only libcuda shim"
LABEL org.opencontainers.image.source="https://github.com/lupinemachines/lupine"
LABEL org.opencontainers.image.version="${CUDA_VERSION}-ubuntu${UBUNTU_VERSION}-slim"

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

RUN ln -sf /opt/lupine/lib/libcuda.so.1 /opt/lupine/lib/libcuda.so \
    && ln -sf /opt/lupine/lib/libnvidia-ml.so.1 /opt/lupine/lib/libnvidia-ml.so

ENV LUPINE_LIBCUDA=/opt/lupine/lib/libcuda.so.1
ENV LUPINE_LIB=/opt/lupine/lib/libcuda.so.1
ENV LD_LIBRARY_PATH=/opt/lupine/lib

ENTRYPOINT []
CMD ["bash"]

FROM nvidia/cuda:${CUDA_VERSION}-${CUDA_RUNTIME_IMAGE_FLAVOR}-ubuntu${UBUNTU_VERSION} AS server

ARG DEBIAN_FRONTEND=noninteractive
ARG CUDA_VERSION
ARG UBUNTU_VERSION

LABEL org.opencontainers.image.title="lupine-server"
LABEL org.opencontainers.image.description="LUPINE server runtime"
LABEL org.opencontainers.image.source="https://github.com/lupinemachines/lupine"
LABEL org.opencontainers.image.version="${CUDA_VERSION}-ubuntu${UBUNTU_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    libnghttp2-14 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=server-build /opt/lupine/build/lupine_driver_server /opt/lupine/bin/lupine_driver_server

RUN chmod +x /opt/lupine/bin/lupine_driver_server

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

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"

RUN dnf install -y gcc gcc-c++ make cmake perl binutils file tar gzip \
    && if [ -n "${GCC_TOOLSET}" ]; then dnf install -y "${GCC_TOOLSET}-gcc" "${GCC_TOOLSET}-gcc-c++"; fi \
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

# scl_source puts the toolset gcc on PATH for this shell only; cmake inherits.
RUN set -eux; \
    if [ -n "${GCC_TOOLSET}" ]; then . scl_source enable "${GCC_TOOLSET}"; fi; \
    cmake -S /opt/lupine -B /opt/lupine/build-static \
      -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
      -DLUPINE_STATIC_DEPS=ON \
      -DNGHTTP2_INCLUDE_DIR=/opt/static-deps/include \
      -DNGHTTP2_LIBRARY=/opt/static-deps/lib/libnghttp2.a \
      -DOPENSSL_ROOT_DIR=/opt/static-deps \
      -DCMAKE_LIBRARY_PATH="${CUDA_HOME}/lib64/stubs"; \
    cmake --build /opt/lupine/build-static --parallel "$(nproc)" \
      --target lupine_driver lupine_nvml

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

RUN printf 'cuda_version=%s\nmin_glibc=%s\n' \
      "${CUDA_VERSION}" "${MAX_GLIBC}" > /artifacts/metadata \
    && ln -s libcuda.so.1 /artifacts/libcuda.so \
    && ln -s libnvidia-ml.so.1 /artifacts/libnvidia-ml.so

CMD ["sh", "-c", "cp -a /artifacts/. \"${ARTIFACTS_DEST:-/target}/\" && echo copied to ${ARTIFACTS_DEST:-/target}"]
