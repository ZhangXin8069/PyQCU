# dev87 对照环境：优先解析本任务构建的 libquda（LD_LIBRARY_PATH 前缀压过全局旧库）
export QUDA_INSTALL=/tmp/opencode/quda-install
export QUDA_PATH=$QUDA_INSTALL
export LD_LIBRARY_PATH=$QUDA_INSTALL/lib:$LD_LIBRARY_PATH
export DEV87_REDUCE_SYNC=1
