# dev87 对照环境：优先使用仓库 data/quda-sm70-install，否则回退到旧的双精度构建；
# 调用方仍可通过
# 预先设置 QUDA_INSTALL/QUDA_BUILD_DIR 覆盖。LD_LIBRARY_PATH 前缀压过全局旧库。
if [ -n "${BASH_SOURCE[0]-}" ]; then
    DEV87_ENV_FILE=${BASH_SOURCE[0]}
elif [ -n "${ZSH_VERSION:-}" ]; then
    # zsh does not populate BASH_SOURCE; %x is the currently sourced file.
    DEV87_ENV_FILE=$(eval 'print -r -- ${(%):-%x}')
else
    DEV87_ENV_FILE=$0
fi
DEV87_ROOT=$(cd "$(dirname "$DEV87_ENV_FILE")/../../../../.." && pwd)
if [ -z "${QUDA_INSTALL:-}" ]; then
    if [ -f "${DEV87_ROOT}/data/quda-sm70-install/lib/libquda.so" ]; then
        export QUDA_INSTALL=${DEV87_ROOT}/data/quda-sm70-install
    else
        export QUDA_INSTALL=${DEV87_ROOT}/data/quda-double-install
    fi
fi
export QUDA_PATH=$QUDA_INSTALL
export QUDA_BUILD_DIR=${QUDA_BUILD_DIR:-${DEV87_ROOT}/data/quda-qio-build/attempt2}
export LD_LIBRARY_PATH=$QUDA_INSTALL/lib:${LD_LIBRARY_PATH:-}
export DEV87_REDUCE_SYNC=1
unset DEV87_ROOT DEV87_ENV_FILE
