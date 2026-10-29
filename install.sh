#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
MODE=${1:-cuda}

usage() {
    cat <<'EOF'
Usage: bash install.sh [kupl|cuda]

  kupl  Build the independent KUPL C++ library and skip the CUDA/Cython build.
  cuda  Build the CUDA C++ library and Cython extension (default, legacy mode).

KUPL environment variables:
  PYQCU_KUPL_BUILD_DIR  Build directory (default: cpp/kupl/qcu/build)
  PYQCU_BUILD_JOBS      Parallel build jobs
  CMAKE_BUILD_TYPE      CMake build type (default: Release)
  KUPL_ENABLE_NATIVE    ON/OFF local ISA tuning (default: OFF)
  KUPL_ENABLE_LTO       ON/OFF link-time optimization (default: OFF)
  KUPL_BUILD_TESTS      ON/OFF build CTest executables (default: ON)
EOF
}

case "$MODE" in
    kupl)
        build_dir=${PYQCU_KUPL_BUILD_DIR:-"$ROOT/cpp/kupl/qcu/build"}
        build_type=${CMAKE_BUILD_TYPE:-Release}
        native=${KUPL_ENABLE_NATIVE:-OFF}
        lto=${KUPL_ENABLE_LTO:-OFF}
        build_tests=${KUPL_BUILD_TESTS:-ON}
        if [[ -z ${PYQCU_BUILD_JOBS:-} ]]; then
            PYQCU_BUILD_JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')
        fi

        cmake -S "$ROOT/cpp/kupl/qcu" -B "$build_dir" \
            -DCMAKE_BUILD_TYPE="$build_type" \
            -DKUPL_ENABLE_NATIVE="$native" \
            -DKUPL_ENABLE_LTO="$lto" \
            -DKUPL_BUILD_TESTS="$build_tests"
        cmake --build "$build_dir" -j "$PYQCU_BUILD_JOBS"

        kupl_library="$build_dir/libqcu.so"
        if [[ ! -f "$kupl_library" ]]; then
            printf 'error: KUPL library was not produced: %s\n' "$kupl_library" >&2
            exit 1
        fi

        PYQCU_BUILD_BACKEND=kupl \
        PYQCU_KUPL_LIB="$kupl_library" \
        PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            python "$ROOT/setup.py" build_ext --inplace
        PYQCU_KUPL_LIB="$kupl_library" \
        PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            python -B -c \
            'from pyqcu.kupl import qcu; print(f"KUPL library: {qcu.library_path()}")'

        printf 'KUPL installation complete.\n'
        printf '  library: %s\n' "$kupl_library"
        printf '  export PYQCU_KUPL_LIB=%q\n' "$kupl_library"
        ;;
    cuda)
        PYQCU_BUILD_BACKEND=cuda python "$ROOT/setup.py" build_ext --inplace
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
