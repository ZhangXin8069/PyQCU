# dev87 对照环境：默认使用仓库 data/ 中已完成 QIO/QMP 的构建；调用方仍可
# 通过预先设置 QUDA_INSTALL 覆盖。LD_LIBRARY_PATH 前缀压过全局旧库。
if [ -n "${BASH_SOURCE[0]-}" ]; then
    DEV87_ENV_FILE=${BASH_SOURCE[0]}
elif [ -n "${ZSH_VERSION:-}" ]; then
    # zsh does not populate BASH_SOURCE; %x is the currently sourced file.
    DEV87_ENV_FILE=$(eval 'print -r -- ${(%):-%x}')
else
    DEV87_ENV_FILE=$0
fi
DEV87_ROOT=$(cd "$(dirname "$DEV87_ENV_FILE")/../../.." && pwd)
export QUDA_INSTALL=${QUDA_INSTALL:-${DEV87_ROOT}/data/quda-qio-install}
export QUDA_PATH=$QUDA_INSTALL
export LD_LIBRARY_PATH=$QUDA_INSTALL/lib:${LD_LIBRARY_PATH:-}
export DEV87_REDUCE_SYNC=1
unset DEV87_ROOT DEV87_ENV_FILE
