#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
MODE=${1:-build}
BUILD_DIR=${PYQCU_KUPL_BUILD_DIR:-"$SCRIPT_DIR/build"}
BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
NATIVE=${KUPL_ENABLE_NATIVE:-OFF}
LTO=${KUPL_ENABLE_LTO:-OFF}
BUILD_TESTS=${KUPL_BUILD_TESTS:-ON}
JOBS=${PYQCU_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')}
BUILT_LIBRARY="$BUILD_DIR/libqcu.so"
PUBLIC_LIBRARY="$SCRIPT_DIR/libqcu.so"
TEMP_PUBLIC_LIBRARY="${PUBLIC_LIBRARY}.tmp.$$"
trap 'rm -f "$TEMP_PUBLIC_LIBRARY"' EXIT

printf 'PYQCU KUPL backend: ready\n'
printf '  build directory: %s\n' "$BUILD_DIR"
printf '  public library: %s\n' "$PUBLIC_LIBRARY"
printf '  native=%s lto=%s tests=%s jobs=%s\n' \
    "$NATIVE" "$LTO" "$BUILD_TESTS" "$JOBS"

if [[ "$MODE" == "--check" ]]; then
    exit 0
fi
if [[ "$MODE" != "build" ]]; then
    printf 'Usage: bash %s [--check]\n' "${BASH_SOURCE[0]:-$0}" >&2
    exit 2
fi

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DKUPL_ENABLE_NATIVE="$NATIVE" \
    -DKUPL_ENABLE_LTO="$LTO" \
    -DKUPL_BUILD_TESTS="$BUILD_TESTS"
cmake --build "$BUILD_DIR" --parallel "$JOBS"
if [[ ! -f "$BUILT_LIBRARY" ]]; then
    printf 'error: KUPL library was not produced: %s\n' "$BUILT_LIBRARY" >&2
    exit 1
fi
BUILD_DIR_ABSOLUTE=$(realpath "$BUILD_DIR")
LINK_TARGET=$(realpath --relative-to="$SCRIPT_DIR" "$BUILD_DIR_ABSOLUTE")/libqcu.so
ln -s "$LINK_TARGET" "$TEMP_PUBLIC_LIBRARY"
mv -Tf "$TEMP_PUBLIC_LIBRARY" "$PUBLIC_LIBRARY"
trap - EXIT
printf 'KUPL library: %s -> %s\n' "$PUBLIC_LIBRARY" "$LINK_TARGET"
printf 'make.sh: SUCCESS\n'
