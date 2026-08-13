#!/usr/bin/env bash
# dev74 —— 集群（512G 内存 / 32G 显存服务器）大格子运行脚本。
#
# 本地（RTX 4060 8GB）只做小格子验证（mg_dev74_bench.py 默认模式）；
# 本脚本生成并执行集群大格子测量命令序列，按实测校准预算模型分级：
#
#   A 级（cold 构建 + warm 求解均可）：16x32x32x32
#   B 级（仅 warm 可行，cold 构建 OOM —— 需分阶段先构建缓存）：16x32x32x64
#   C 级（单卡不可行，需多卡分布式）：24x32x32x64
#
# 用法：
#   bash examples/qcu/mg_dev74_cluster.sh            # dry-run：只打印命令
#   RUN=1 bash examples/qcu/mg_dev74_cluster.sh      # 实际执行（集群上）
#   bash examples/qcu/mg_dev74_cluster.sh --build-cpp  # 粗算子构建用 C++ dslash
set -euo pipefail

REPO=${HOME}/PyQCU
LOG_DIR="$REPO/logs"
RUN="${RUN:-0}"
BUILD="${1:---build py}"
LAT16="16 32 32 32"
LAT64="16 32 32 64"
LAT24="24 32 32 64"

echo "=== dev74 cluster runner (RUN=$RUN, $BUILD) ==="

# ---------- A 级：16x32x32x32（cold+warm 均可行，~28GB cold / ~14GB warm）----------
run_one() { # $1=label  $2..$5=lattice
  local label=$1; shift
  local cmd="python $REPO/examples/qcu/mg_dev74_clean.py --lattice $1 $2 $3 $4 \
--prec c64 --levels 2 --restart 10 --ct 1e5 --cmi 15 --pairs 5 $BUILD"
  if [ "$RUN" = "1" ]; then
    echo ">>> $cmd"
    (cd "$REPO" && source ./env.sh >/dev/null 2>&1 && $cmd)
  else
    echo "[dry-run] $cmd"
  fi
}

echo "--- A 级（cold 构建 ~28GB + warm 求解 ~14GB，32G 显存可行）---"
run_one 16x32x32x32 $LAT16
run_one 16x32x32x64 $LAT64
run_one 24x32x32x64 $LAT24

echo "--- 汇总与报告 ---"
summary="python $REPO/examples/qcu/mg_dev74_collect.py && \
python $REPO/examples/qcu/mg_dev74_budget.py --fit --mode cluster && \
python $REPO/examples/qcu/mg_dev74_mktable.py && \
python $REPO/examples/qcu/mg_dev74_plots.py"
if [ "$RUN" = "1" ]; then
  echo ">>> $summary"
  (cd "$REPO" && source ./env.sh >/dev/null 2>&1 && bash -c "$summary")
else
  echo "[dry-run] $summary"
fi

echo "=== 说明 ==="
echo " * 16x32x32x64 cold 构建预测 ~57GB（超 32G）→ 集群上首次运行若 OOM，"
echo "   先单独构建缓存（--build cpp 降低峰值），再跑 warm 测量（~28GB 可行）"
echo " * 24x32x32x64 cold ~85GB / warm ~42GB，单卡不可行 → 需多卡分布式（MPI）"
echo " * 每次测量为独立进程；nullvec 缓存复用（logs/nullvec_cache）"
