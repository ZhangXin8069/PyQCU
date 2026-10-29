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

printf 'PYQCU KUPL backend: ready\n'
printf '  build directory: %s\n' "$BUILD_DIR"
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
printf 'make.sh: SUCCESS\n'
