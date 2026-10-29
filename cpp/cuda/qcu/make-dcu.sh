#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
MODE=${1:-build}
BUILD_DIR=${QCU_DCU_BUILD_DIR:-"$REPO_ROOT/data/build/cuda-qcu-dcu"}
BUILD_TYPE=${QCU_DCU_BUILD_TYPE:-Release}
JOBS=${PYQCU_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')}

resolve_executable() {
    local candidate=$1
    if [[ "$candidate" == */* ]]; then
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
        fi
        return
    fi
    command -v "$candidate" 2>/dev/null || true
}

compiler=${QCU_DCU_CUDA_COMPILER:-}
if [[ -n "$compiler" ]]; then
    compiler=$(resolve_executable "$compiler")
elif [[ -n ${ROCM_PATH:-} && -x "$ROCM_PATH/cuda/bin/nvcc" ]]; then
    compiler="$ROCM_PATH/cuda/bin/nvcc"
fi

if [[ -z "$compiler" || ! -x "$compiler" ]]; then
    printf '%s\n' \
        'error: no DTK nvcc-compatible compiler found; set QCU_DCU_CUDA_COMPILER' \
        'or ROCM_PATH with ROCM_PATH/cuda/bin/nvcc' >&2
    exit 2
fi

printf 'PYQCU DCU backend: ready\n'
printf '  compiler: %s\n' "$compiler"
printf '  build directory: %s\n' "$BUILD_DIR"

if [[ "$MODE" == "--check" ]]; then
    exit 0
fi
if [[ "$MODE" != "build" ]]; then
    printf 'Usage: bash %s [--check]\n' "${BASH_SOURCE[0]:-$0}" >&2
    exit 2
fi

ln -sf CMakeLists-dcu.txt "$SCRIPT_DIR/CMakeLists.txt"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_CUDA_COMPILER="$compiler" \
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="$SCRIPT_DIR"
cmake --build "$BUILD_DIR" --parallel "$JOBS"
printf 'make-dcu.sh: SUCCESS\n'
