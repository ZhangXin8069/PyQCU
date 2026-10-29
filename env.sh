#!/usr/bin/env bash
_PYQCU_ENV_ROOT=$(
    cd "$(dirname "${BASH_SOURCE[0]:-$0}")"
    pwd
)
_PYQCU_ENV_NAME=$(basename "${BASH_SOURCE[0]:-$0}")
echo "###${_PYQCU_ENV_NAME} in ${_PYQCU_ENV_ROOT} is sourcing...:$(date "+%Y-%m-%d-%H-%M-%S")###"
export PYQCU_ROOT=${_PYQCU_ENV_ROOT}

_pyqcu_prepend_path() {
    local name="$1"
    local first="$2"
    local current="$3"
    local entry
    local remainder="$current"
    local result="$first"

    while [ -n "$remainder" ]; do
        case "$remainder" in
            *:*) entry=${remainder%%:*}; remainder=${remainder#*:} ;;
            *) entry=$remainder; remainder= ;;
        esac
        [ -n "$entry" ] || continue
        [ "$entry" != "$first" ] || continue
        case ":${result}:" in
            *":${entry}:"*) ;;
            *) result="${result}:${entry}" ;;
        esac
    done

    case "$name" in
        PYTHONPATH) export PYTHONPATH="$result" ;;
        LD_LIBRARY_PATH) export LD_LIBRARY_PATH="$result" ;;
        *)
            printf 'error: unsupported path variable: %s\n' "$name" >&2
            return 2
            ;;
    esac
}

_PYQCU_ENV_QUDA_INSTALL=${QUDA_INSTALL:-${_PYQCU_ENV_ROOT}/data/quda-sm70-install}
if [ -f "${_PYQCU_ENV_QUDA_INSTALL}/lib/libquda.so" ]; then
    export QUDA_INSTALL=${_PYQCU_ENV_QUDA_INSTALL}
    export QUDA_PATH=${_PYQCU_ENV_QUDA_INSTALL}
    _pyqcu_prepend_path LD_LIBRARY_PATH "${_PYQCU_ENV_ROOT}/cpp/cuda/qcu" "${LD_LIBRARY_PATH-}"
    _pyqcu_prepend_path LD_LIBRARY_PATH "${_PYQCU_ENV_QUDA_INSTALL}/lib" "${LD_LIBRARY_PATH-}"
else
    _pyqcu_prepend_path LD_LIBRARY_PATH "${_PYQCU_ENV_ROOT}/cpp/cuda/qcu" "${LD_LIBRARY_PATH-}"
fi

# Keep this checkout ahead of stale editable/install-tree PYTHONPATH entries.
_pyqcu_prepend_path PYTHONPATH "${_PYQCU_ENV_ROOT}" "${PYTHONPATH-}"

# Canonical library locations. Explicit caller overrides always win.
export PYQCU_CUDA_LIB=${PYQCU_CUDA_LIB:-${_PYQCU_ENV_ROOT}/cpp/cuda/qcu/libqcu.so}
export QCU_DCU_BUILD_DIR=${QCU_DCU_BUILD_DIR:-${_PYQCU_ENV_ROOT}/data/build/cuda-qcu-dcu}
export PYQCU_DCU_LIB=${PYQCU_DCU_LIB:-${QCU_DCU_BUILD_DIR}/libqcu.so}
export PYQCU_KUPL_BUILD_DIR=${PYQCU_KUPL_BUILD_DIR:-${_PYQCU_ENV_ROOT}/cpp/kupl/qcu/build}
export PYQCU_KUPL_LIB=${PYQCU_KUPL_LIB:-${_PYQCU_ENV_ROOT}/cpp/kupl/qcu/libqcu.so}
export QCU_KUPL_VARIANT=${QCU_KUPL_VARIANT:-${PYQCU_KUPL_LIB}}

# Conservative KUPL defaults; scheduling and memory-node choices remain explicit.
export OMP_DYNAMIC=${OMP_DYNAMIC:-FALSE}
export KUPL_NUMA_BIND=${KUPL_NUMA_BIND:-0}
export KUPL_FUSED_BICG_DOTS=${KUPL_FUSED_BICG_DOTS:-1}
export KUPL_DISABLE_CLOVER_CACHE=${KUPL_DISABLE_CLOVER_CACHE:-0}
export KUPL_ALLOW_OVERSUBSCRIBE=${KUPL_ALLOW_OVERSUBSCRIBE:-0}
export KUPL_HUGEPAGE=${KUPL_HUGEPAGE:-0}

if [ -n "${KUPL_OMP_THREADS:-}" ] && [ -z "${OMP_NUM_THREADS:-}" ]; then
    export OMP_NUM_THREADS=${KUPL_OMP_THREADS}
elif [ -z "${KUPL_OMP_THREADS:-}" ] && [ -n "${OMP_NUM_THREADS:-}" ]; then
    export KUPL_OMP_THREADS=${OMP_NUM_THREADS}
fi

# Library-owned binding cannot coexist with OpenMP runtime placement.
if [ "${KUPL_NUMA_BIND}" = "1" ]; then
    unset OMP_PROC_BIND OMP_PLACES GOMP_CPU_AFFINITY KMP_AFFINITY
fi

export MPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
echo "###${_PYQCU_ENV_NAME} in ${_PYQCU_ENV_ROOT} is done......:$(date "+%Y-%m-%d-%H-%M-%S")###"
unset -f _pyqcu_prepend_path
unset _PYQCU_ENV_QUDA_INSTALL _PYQCU_ENV_NAME _PYQCU_ENV_ROOT
