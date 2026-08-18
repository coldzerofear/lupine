#!/usr/bin/env bash
#
# Verify a LUPINE_STATIC_DEPS=ON server binary is actually self-contained.
#
# The static server exists so the image can drop the CUDA runtime base: once
# nghttp2 and the C++ runtime are linked in, the only libraries left are glibc
# and the ones the container runtime injects from the host driver. Every check
# here guards one way that promise silently breaks:
#
#   1. ldd allowlist     - a stray DT_NEEDED (libnghttp2, libstdc++, libgcc_s,
#                          libcudart) is a library the runtime image would have
#                          to ship, which is exactly what the minimal base does
#                          not have. glibc family + the driver libraries only.
#   2. driver linkage    - libcuda.so.1 MUST be present. It is the one dependency
#                          deliberately left dynamic (nvidia-container-runtime
#                          injects the host's copy); if it went missing, the
#                          build linked something other than the CUDA driver and
#                          the server would fail at cuInit instead of at build.
#   3. glibc ceiling     - the binary runs only on images whose glibc is >= the
#                          max version referenced. The ceiling is an input (per
#                          builder base); exceeding it means the build ran on a
#                          newer base than the artifact claims to support.
#
# There is deliberately no .dynsym check (an executable exports nothing that can
# interpose) and no TLS check (the server is plaintext by design -- front it
# with a TLS proxy; only the client links OpenSSL).
#
# Usage: check_static_server.sh <lupine_driver_server> <max-glibc>
#   e.g. check_static_server.sh build-static/lupine_driver_server 2.28

set -o errexit
set -o nounset
set -o pipefail

SERVER="${1:?usage: $0 <lupine_driver_server> <max-glibc>}"
MAX_GLIBC="${2:?missing max glibc version (e.g. 2.28)}"

fail=0

test -f "$SERVER" || { echo "[FAIL] missing artifact: $SERVER"; exit 1; }
test -x "$SERVER" || { echo "[FAIL] not executable: $SERVER"; exit 1; }

# DT_NEEDED allowlist: the glibc family, the dynamic loader, and the driver
# libraries the container runtime injects. libgcc_s and libstdc++ are
# deliberately absent: -static-libgcc/-static-libstdc++ must have removed them.
ALLOWED_RE='^(libc\.so|libm\.so|libpthread\.so|libdl\.so|librt\.so|ld-linux|linux-vdso|libcuda\.so|libnvidia-ml\.so)'

needed=$(readelf -d "$SERVER" | awk '/NEEDED/{gsub(/[\[\]]/,"",$5); print $5}')

bad=$(echo "$needed" | grep -vE "${ALLOWED_RE}" || true)
if [[ -n "$bad" ]]; then
  echo "[FAIL] $SERVER has DT_NEEDED entries the minimal runtime image cannot satisfy:"
  echo "$bad" | sed 's/^/         /'
  fail=1
else
  echo "[ok] $SERVER DT_NEEDED confined to glibc + injected driver libraries"
fi

if echo "$needed" | grep -qE '^libcuda\.so'; then
  echo "[ok] $SERVER links the CUDA driver (injected at runtime)"
else
  echo "[FAIL] $SERVER does not link libcuda.so.1 -- CUDA::cuda_driver did not resolve"
  fail=1
fi

# Highest GLIBC_x.y version referenced must not exceed the declared ceiling.
max=$(readelf -V "$SERVER" 2>/dev/null | grep -oE 'GLIBC_[0-9]+\.[0-9]+' \
      | sort -uV | tail -1 | sed 's/GLIBC_//')
if [[ -z "$max" ]]; then
  echo "[FAIL] $SERVER: could not determine referenced glibc versions"
  fail=1
elif printf '%s\n%s\n' "$max" "$MAX_GLIBC" | sort -V | tail -1 | grep -qx "$MAX_GLIBC"; then
  echo "[ok] $SERVER glibc ceiling $max <= $MAX_GLIBC"
else
  echo "[FAIL] $SERVER references GLIBC_$max > declared ceiling $MAX_GLIBC"
  fail=1
fi

if (( fail )); then
  echo "static server verification FAILED"
  exit 1
fi
echo "static server verification passed"