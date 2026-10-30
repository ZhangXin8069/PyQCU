#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
MODE=${1:-cuda}

usage() {
    cat <<'EOF'
Usage: bash install.sh [kupl|cuda|dcu]

  kupl  Build the independent KUPL C++ library and skip the CUDA/Cython build.
  cuda  Build the CUDA C++ library and Cython extension (default, legacy mode).
  dcu   Build the DTK/ROCm-compatible GPU library and Cython extension.

KUPL environment variables:
  PYQCU_KUPL_BUILD_DIR  Build directory (default: cpp/kupl/qcu/build)
  PYQCU_KUPL_LIB        Canonical KUPL library path exported by env.sh
  PYQCU_BUILD_JOBS      Parallel build jobs
  CMAKE_BUILD_TYPE      CMake build type (default: Release)
  KUPL_ENABLE_NATIVE    ON/OFF local ISA tuning (default: OFF)
  KUPL_ENABLE_LTO       ON/OFF link-time optimization (default: OFF)
  KUPL_BUILD_TESTS      ON/OFF build CTest executables (default: ON)

DCU environment variables:
  QCU_DCU_CUDA_COMPILER  DTK nvcc-compatible compiler path override
  QCU_DCU_BUILD_DIR      Build directory (default: data/build/cuda-qcu-dcu)
  PYQCU_DCU_LIB          Canonical DCU library path exported by env.sh
  ROCM_PATH              Used to locate ${ROCM_PATH}/cuda/bin/nvcc

CUDA library variable:
  PYQCU_CUDA_LIB         Canonical CUDA library path exported by env.sh
EOF
}

case "$MODE" in
    kupl)
        build_dir=${PYQCU_KUPL_BUILD_DIR:-"$ROOT/cpp/kupl/qcu/build"}
        build_type=${CMAKE_BUILD_TYPE:-Release}
        native=${KUPL_ENABLE_NATIVE:-OFF}
        lto=${KUPL_ENABLE_LTO:-OFF}
        build_tests=${KUPL_BUILD_TESTS:-ON}
        PYQCU_KUPL_BUILD_DIR="$build_dir" \
        CMAKE_BUILD_TYPE="$build_type" \
        KUPL_ENABLE_NATIVE="$native" \
        KUPL_ENABLE_LTO="$lto" \
        KUPL_BUILD_TESTS="$build_tests" \
            bash "$ROOT/cpp/kupl/qcu/make.sh"

        kupl_library="$build_dir/libqcu.so"
        if [[ ! -f "$kupl_library" ]]; then
            printf 'error: KUPL library was not produced: %s\n' "$kupl_library" >&2
            exit 1
        fi
        kupl_public_library="$ROOT/cpp/kupl/qcu/libqcu.so"
        if [[ ! -f "$kupl_public_library" ]]; then
            printf 'error: KUPL public library symlink was not produced: %s\n' \
                "$kupl_public_library" >&2
            exit 1
        fi

        PYQCU_BUILD_BACKEND=kupl \
        PYQCU_KUPL_LIB="$kupl_public_library" \
        PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            python "$ROOT/setup.py" build_ext --inplace
        PYQCU_KUPL_LIB="$kupl_public_library" \
        PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            python -B -c \
            'from pyqcu.kupl import qcu; print(f"KUPL library: {qcu.library_path()}")'

        printf 'KUPL installation complete.\n'
        printf '  built library: %s\n' "$kupl_library"
        printf '  public library: %s\n' "$kupl_public_library"
        printf '  export PYQCU_KUPL_LIB=%q\n' "$kupl_public_library"
        ;;
    cuda)
        PYQCU_BUILD_BACKEND=cuda python "$ROOT/setup.py" build_ext --inplace
        rm -rf "$ROOT/build"
        ;;
    dcu)
        PYQCU_BUILD_BACKEND=dcu python "$ROOT/setup.py" build_ext --inplace
        rm -rf "$ROOT/build"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
