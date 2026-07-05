#!/usr/bin/env bash
_PATH=$(
    cd "$(dirname "$0")"
    pwd
)
_NAME=$(basename "$0")
echo "###${_NAME} in ${_PATH} is sourcing...:$(date "+%Y-%m-%d-%H-%M-%S")###"
export LD_LIBRARY_PATH=${_PATH}/cpp/cuda/qcu:$LD_LIBRARY_PATH
export PYTHONPATH=${_PATH}:${PYTHONPATH}
export MPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
echo "###${_NAME} in ${_PATH} is done......:$(date "+%Y-%m-%d-%H-%M-%S")###"


