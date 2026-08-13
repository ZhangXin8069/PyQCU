#!/usr/bin/env bash
# dev74_1 —— 服务器（512G 内存 / 32G 显存）MG 加速比验证一键流程。
#
# 目标：确保 MG 相对 BiStabCG 的加速比 > 1.5（--gate，默认 1.5）。
# 依据：dev73_5 在 V100-32G 实测 8x8x8x16 speedup=2.43x；本流程以
#   Step 1 为强制闸门：8x8x8x16 达标（预期 ~2.4x）才继续后续步骤。
#
# 用法：
#   bash examples/qcu/mg_dev74_1_server.sh            # dry-run：打印命令
#   RUN=1 bash examples/qcu/mg_dev74_1_server.sh      # 实际执行（服务器上）
#   GATE=1.5 可选（默认 1.5）
set -euo pipefail

REPO=/root/PyQCU
LOG_DIR="$REPO/logs"
RUN="${RUN:-0}"
GATE="${GATE:-1.5}"
PY="python"
CLEAN="$REPO/examples/qcu/mg_dev74_clean.py"
SWEEP="$REPO/examples/qcu/mg_dev74_1_sweep.py"
CHECK="$REPO/examples/qcu/mg_dev74_1_check.py"

step() { echo; echo "===== $1 ====="; }

run() { # $@ command
  if [ "$RUN" = "1" ]; then
    echo ">>> $*"
    (cd "$REPO" && source ./env.sh >/dev/null 2>&1 && "$@")
  else
    echo "[dry-run] $*"
  fi
}

step "Step 0: 环境自检"
run nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run $PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

step "Step 1: 快速验证（强制闸门）—— 8x8x8x16 c64 2L，预期 speedup ~2.4x >= ${GATE}"
run $PY "$CLEAN" --lattice 8 8 8 16 --prec c64 --levels 2 --restart 10 --ct 1e5 --cmi 15 --pairs 5
if [ "$RUN" = "1" ]; then
  GATE_FILE=$(ls -t "$LOG_DIR"/dev74_clean_L8x8x8x16*.json 2>/dev/null | head -1)
  if [ -n "$GATE_FILE" ] && $PY "$CHECK" --gate "$GATE" --label "Step1 gate" --file "$GATE_FILE"; then
    echo "Step 1 通过：speedup >= ${GATE}，继续"
  else
    echo "Step 1 失败：speedup < ${GATE} 或无数据，停止。请检查 GPU/构建/日志后重试。" >&2
    exit 1
  fi
fi

step "Step 2: 参数优化扫描 —— 8x16x16x16（9 配置，推荐 3L / r=20）"
run $PY "$SWEEP" --lattice 8 16 16 16 --pairs 3

step "Step 3: 大格子规模测试 —— 16x32x32x32（cold 28.4GB / warm 14.1GB，32G 单卡可行）"
run $PY "$CLEAN" --lattice 16 32 32 32 --prec c64 --levels 2 --restart 10 --ct 1e5 --cmi 15 --pairs 3

step "Step 4: 大格子 warm 测试 —— 16x32x32x64（warm 28.2GB；若 cold OOM 先 --build cpp）"
run $PY "$CLEAN" --lattice 16 32 32 64 --prec c64 --levels 2 --restart 10 --ct 1e5 --cmi 15 --pairs 3 --build cpp

step "Step 5: 汇总断言"
run $PY "$CHECK" --gate "$GATE" --label "dev74_1 server final" \
  --file "$LOG_DIR/dev74_1_sweep.json"

step "完成。结果：logs/dev74_1_sweep.json + dev74_clean_L*.json"
echo "说明：24x32x32x64（warm 42.3GB）单卡不可行，需多卡 MPI 分布式（后续工作）。"
