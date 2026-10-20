#!/usr/bin/env bash
_PATH=$(
    cd "$(dirname "${BASH_SOURCE[0]:-$0}")"
    pwd
)
_NAME=$(basename "${BASH_SOURCE[0]:-$0}")
echo "###${_NAME} in ${_PATH} is sourcing...:$(date "+%Y-%m-%d-%H-%M-%S")###"
_QUDA_INSTALL=${QUDA_INSTALL:-${_PATH}/data/quda-sm70-install}
if [ -f "${_QUDA_INSTALL}/lib/libquda.so" ]; then
    export QUDA_INSTALL=${_QUDA_INSTALL}
    export QUDA_PATH=${_QUDA_INSTALL}
    export LD_LIBRARY_PATH=${_QUDA_INSTALL}/lib:${_PATH}/cpp/cuda/qcu:${LD_LIBRARY_PATH}
else
    export LD_LIBRARY_PATH=${_PATH}/cpp/cuda/qcu:${LD_LIBRARY_PATH}
fi
export PYTHONPATH=${_PATH}:${PYTHONPATH}
export MPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
unset _QUDA_INSTALL
echo "###${_NAME} in ${_PATH} is done......:$(date "+%Y-%m-%d-%H-%M-%S")###"

