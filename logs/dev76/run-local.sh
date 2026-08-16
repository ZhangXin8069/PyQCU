#!/usr/bin/env bash
# test76 —— 多线程版（一线程一卡）CUDA C++ MultiGrid 求解器测试套件运行脚本。
#
# 设备分配（本机 3 卡，CUDA 运行时视角 0=V100-32GB, 1/2=P100-16GB×2）：
#   * 多线程测试：2 线程 × P100×2（device_ids=[1,2]）—— 主测试对象
#   * 单线程大格子：V100（device_ids=[0]）
#   * 三卡并行不测（任务约束）
#
# 流程：verify → clean(P100×2 8x8x8x16) → bench（P100×2 + V100 单线程大格子）
#       → sweep(P100×2 参数扫描) → check(gate) → collect/mktable/plots
#       → budget(16G P100 / 32G V100) → 归档收敛日志。
#
# 数据持久化全部 h5py（main.py 内 save_dict_h5/load_dict_h5，独立 File 句柄
# 多线程安全）；PNG/TeX 为图表展示产物。
#
# 用法：
#   bash logs/test76/run-local.sh            # 实际执行
#   bash logs/test76/run-local.sh --dry-run  # 只打印命令
set -uo pipefail

REPO="${HOME}/PyQCU"
WORK="$REPO/logs/test76"
MAIN="$WORK/main.py"
TS=$(date +%Y%m%d-%H%M%S)

# ---- 版本目录：v<YYYYMMDDHHMM>；同分钟重跑加 -<SS> 防覆盖 ----
VDIR="$WORK/v$(date +%Y%m%d%H%M)"
if [ -e "$VDIR" ]; then VDIR="$VDIR-$(date +%S)"; fi
mkdir -p "$VDIR"
export TEST76_OUTDIR="$VDIR"
LOG_FILE="$VDIR/run-local-$TS.log"

DRY="${DRY:-${1:-}}"
[ "$DRY" = "--dry-run" ] && DRY=1 || DRY=0

step() { echo; echo "===== $1 ====="; }
run() { # $1=timeout_s  "$@"=command
  local tmo="$1"; shift
  if [ "$DRY" = "1" ]; then echo "[dry-run] timeout $tmo $*"; return 0; fi
  echo ">>> timeout $tmo $*"
  timeout "$tmo" "$@"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[warn] 上一步失败 rc=$rc（继续）" >&2
  fi
  return 0
}

{
echo "=== test76 run-local @ $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "VDIR=$VDIR"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

# Step 0 自检：构建产物与扩展存在
step "Step 0 自检"
run 60 bash -c "test -f $REPO/cpp/cuda/qcu/libqcu.so && \
  ls $REPO/pyqcu/cuda/qcu*.so >/dev/null && echo 'libqcu.so + cython ext OK'"

# Step 1 正确性验证（一致性 P100×2 + 独立问题 + V100 单线程 + h5py 多线程 IO）
step "Step 1 verify（多线程正确性 + h5py IO）"
run 1800 python "$MAIN" verify

# Step 2 干净测量：P100×2 多线程 8x8x8x16 2L
step "Step 2 clean（P100×2 8x8x8x16 2L）"
run 1200 python "$MAIN" clean --lattice 8 8 8 16 --levels 2 --restart 5 \
    --ct 1e5 --cmi 15 --nthreads 2 --devices 1 2

# Step 3 批量基准（P100×2 多线程 + V100 单线程大格子）
step "Step 3 bench"
run 7200 python "$MAIN" bench --pairs 3

# Step 4 参数扫描（P100×2 8x8x8x16：r/ct/cmi/levels × speedup）
step "Step 4 sweep（P100×2 8x8x8x16）"
run 10800 python "$MAIN" sweep --lattice 8 8 8 16

# Step 5 加速比断言（sweep 半数配置 ≥ gate=1.5）
step "Step 5 check（gate=1.5）"
run 300 python "$MAIN" check --gate 1.5

# Step 6 汇总/表/图
step "Step 6 collect + mktable + plots"
run 300 python "$MAIN" collect
run 300 python "$MAIN" mktable
run 300 python "$MAIN" plots

# Step 7 显存/内存预算（16G P100 档 + 32G V100 档）
step "Step 7 budget（16G / 32G）"
run 300 python "$MAIN" budget --vram 16
run 300 python "$MAIN" budget --vram 32

# Step 8 归档 C++ 收敛日志
step "Step 8 归档 clover_multigrid.log"
if [ -f "$REPO/logs/clover_multigrid.log" ]; then
  cp "$REPO/logs/clover_multigrid.log" "$VDIR/" && echo "copied"
else
  echo "[warn] no clover_multigrid.log"
fi

step "=== done @ $(date '+%Y-%m-%d %H:%M:%S') ==="
ls -la "$VDIR"
} 2>&1 | tee "$LOG_FILE"
