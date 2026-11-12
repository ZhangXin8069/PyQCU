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
# Native backends target Linux only, so CMake always emits libqcu.so.
PLATFORM_LIB_NAME=libqcu.so
BUILT_LIBRARY="$BUILD_DIR/$PLATFORM_LIB_NAME"
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
# Keep absolute path resolution on plain POSIX shell commands; `mv -Tf` is
# GNU-only and is avoided so the staging rename stays portable.
BUILD_DIR_ABSOLUTE=$(cd "$BUILD_DIR" && pwd -P)
case "$BUILD_DIR_ABSOLUTE" in
    "$SCRIPT_DIR"/*) LINK_TARGET=${BUILD_DIR_ABSOLUTE#"$SCRIPT_DIR"/}/"$PLATFORM_LIB_NAME" ;;
    *) LINK_TARGET=$BUILD_DIR_ABSOLUTE/"$PLATFORM_LIB_NAME" ;;
esac
ln -s "$LINK_TARGET" "$TEMP_PUBLIC_LIBRARY"
rm -f "$PUBLIC_LIBRARY"
mv "$TEMP_PUBLIC_LIBRARY" "$PUBLIC_LIBRARY"
trap - EXIT
printf 'KUPL library: %s -> %s\n' "$PUBLIC_LIBRARY" "$LINK_TARGET"
printf 'make.sh: SUCCESS\n'
