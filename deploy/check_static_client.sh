#!/usr/bin/env bash
#
# Verify a LUPINE_STATIC_DEPS=ON client shim bundle is actually self-contained.
#
# The static build exists so the shims can be dropped into arbitrary container
# images. Every check here guards one way that promise silently breaks:
#
#   1. ldd allowlist     - a stray DT_NEEDED (libnghttp2, libssl, libstdc++, or
#                          a lupine artifact absent from this bundle) reintroduces
#                          a dependency the target image would have to satisfy.
#                          The root shims (libcuda.so.1, libnvidia-ml.so.1) each
#                          own an RPC connection and must depend on glibc alone;
#                          every other shim forwards through one of those over
#                          its $ORIGIN rpath, so it may also depend on another
#                          file in this same bundle.
#   2. .dynsym export    - embedded nghttp2/OpenSSL/libstdc++ symbols leaking
#                          into .dynsym would interpose on the target image's
#                          own copies (the version scripts must hide them).
#                          Checked on every shim: none of them should carry
#                          these regardless of tier.
#   3. TLS presence      - find_package(OpenSSL QUIET) degrades SILENTLY when
#                          OpenSSL is missing on the builder; a root shim then
#                          refuses https:// endpoints at runtime. Static
#                          libcrypto embeds its version string, so its absence
#                          is detectable here rather than in production. Only
#                          the root shims carry their own copy of the transport
#                          (and so TLS support); the rest forward through them.
#   4. glibc ceiling     - the artifact loads only on images whose glibc is >=
#                          the max version referenced. Checked on every shim.
#
# Usage: check_static_client.sh <max-glibc> <root.so>... -- <dependent.so>...
#   e.g. check_static_client.sh 2.28 build/libcuda.so.1 build/libnvidia-ml.so.1 \
#          -- build/libnccl.so.2

set -o errexit
set -o nounset
set -o pipefail

MAX_GLIBC="${1:?usage: $0 <max-glibc> <root.so>... -- <dependent.so>...}"
shift

roots=()
while [[ $# -gt 0 && "$1" != "--" ]]; do
  roots+=("$1")
  shift
done
if [[ $# -eq 0 ]]; then
  echo "usage: $0 <max-glibc> <root.so>... -- <dependent.so>..." >&2
  exit 1
fi
shift # the --
dependents=("$@")

if [[ ${#roots[@]} -eq 0 ]]; then
  echo "[FAIL] no root shims given" >&2
  exit 1
fi

fail=0

# SONAMEs of every shim in this bundle, so a dependent shim's DT_NEEDED on
# another one of them is recognized rather than flagged as foreign.
bundle_sonames=()
for so in "${roots[@]}" "${dependents[@]}"; do
  bundle_sonames+=("$(basename "$so")")
done
bundle_re="$(IFS='|'; echo "${bundle_sonames[*]}")"
bundle_re="${bundle_re//./\\.}"

GLIBC_RE='^(libc\.so|libm\.so|libpthread\.so|libdl\.so|librt\.so|ld-linux|linux-vdso)'

check_needed() {
  local so="$1" allow_bundle="$2" bad
  bad=$(readelf -d "$so" | awk '/NEEDED/{gsub(/[\[\]]/,"",$5); print $5}' \
        | grep -vE "${GLIBC_RE}" || true)
  if [[ "$allow_bundle" == 1 && -n "$bad" ]]; then
    bad=$(echo "$bad" | grep -vE "^(${bundle_re})$" || true)
  fi
  if [[ -n "$bad" ]]; then
    echo "[FAIL] $so has DT_NEEDED entries outside its allowance:"
    echo "$bad" | sed 's/^/         /'
    fail=1
  elif [[ "$allow_bundle" == 1 ]]; then
    echo "[ok] $so DT_NEEDED confined to glibc + this bundle"
  else
    echo "[ok] $so DT_NEEDED confined to glibc"
  fi
}

# Exported symbols must stay within the shim's contract. Embedded dependency
# symbols showing up here means the version script no longer covers them.
check_exports() {
  local so="$1"
  local bad
  bad=$(nm -D --defined-only "$so" | awk '{print $3}' \
        | grep -E '^(nghttp2_|SSL_|EVP_|CRYPTO_|BIO_|X509_|_ZSt|_ZNSt|LZ4_)' || true)
  if [[ -n "$bad" ]]; then
    echo "[FAIL] $so leaks embedded dependency symbols into .dynsym:"
    echo "$bad" | head -10 | sed 's/^/         /'
    fail=1
  else
    echo "[ok] $so .dynsym free of embedded dependency symbols"
  fi
}

# TLS presence. Static linking pulls only referenced objects, so libcrypto's
# version STRING may be absent even in a TLS build -- detect by what a TLS
# build cannot avoid dragging in: SSL_* code in .symtab (primary; the version
# script makes them local, but they stay in .symtab unless stripped) or, for
# stripped artifacts, libcrypto's error-format string (in objects the SSL code
# always references).
check_tls() {
  local so="$1"
  # grep -c rather than -q: -q exits on first match, the writer takes SIGPIPE,
  # and under pipefail the whole pipeline reads as failure.
  if [[ "$(nm "$so" 2>/dev/null | grep -cE ' (t|T) SSL_connect')" -gt 0 \
     || "$(strings "$so" | grep -c 'OpenSSL internal error')" -gt 0 ]]; then
    echo "[ok] $so contains statically linked OpenSSL (TLS available)"
  else
    echo "[FAIL] $so built WITHOUT TLS (find_package(OpenSSL QUIET) degraded silently)"
    fail=1
  fi
}

# Highest GLIBC_x.y version referenced must not exceed the declared ceiling.
check_glibc_ceiling() {
  local so="$1"
  local max
  # grep finding nothing is expected (a shim with no versioned glibc symbol) and
  # must not abort the whole script under pipefail -- there are more shims to
  # check after this one fails its own report below.
  max=$(readelf -V "$so" 2>/dev/null | grep -oE 'GLIBC_[0-9]+\.[0-9]+' \
        | sort -uV | tail -1 | sed 's/GLIBC_//' || true)
  if [[ -z "$max" ]]; then
    echo "[FAIL] $so: could not determine referenced glibc versions"
    fail=1
    return
  fi
  if printf '%s\n%s\n' "$max" "$MAX_GLIBC" | sort -V | tail -1 | grep -qx "$MAX_GLIBC"; then
    echo "[ok] $so glibc ceiling $max <= $MAX_GLIBC"
  else
    echo "[FAIL] $so references GLIBC_$max > declared ceiling $MAX_GLIBC"
    fail=1
  fi
}

for so in "${roots[@]}"; do
  test -f "$so" || { echo "[FAIL] missing artifact: $so"; exit 1; }
  check_needed "$so" 0
  check_exports "$so"
  check_tls "$so"
  check_glibc_ceiling "$so"
done

for so in "${dependents[@]}"; do
  test -f "$so" || { echo "[FAIL] missing artifact: $so"; exit 1; }
  check_needed "$so" 1
  check_exports "$so"
  check_glibc_ceiling "$so"
done

if (( fail )); then
  echo "static client verification FAILED"
  exit 1
fi
echo "static client verification passed"
