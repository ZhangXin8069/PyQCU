#!/usr/bin/env bash
# test12 —— 本地运行脚本（RTX 4060 8GB 小卡验证）。
#
# 流程：verify → clean 8x8x8x16 → bench local → sweep 8x8x8x16
#       → collect/mktable/plots/plots1 → 归档收敛日志。
# 特点：
#   * 完整终端输出（tee 归档到 logs/test12/run-local-<ts>.log）
#   * 每步 timeout 防卡壳；单步失败仅记录并继续（GRID 保持轻量）
#
# 用法：
#   bash logs/test12/run-local.sh            # 实际执行
#   bash logs/test12/run-local.sh --dry-run  # 只打印命令
set -uo pipefail

REPO="${HOME}/PyQCU"
WORK="$REPO/logs/test12"
MAIN="$WORK/main.py"
TS=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$WORK/run-local-$TS.log"
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
  return $rc
}

# ---- 环境与完整输出归档 ----
cd "$REPO" || exit 1
source ./env.sh >/dev/null 2>&1
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== test12 local runner start $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

step "Step 0: 环境自检"
run 120 python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
run 120 python -c "from pyqcu.cuda import qcu; print('qcu bridge OK')"

step "Step 1: 正确性验证（8x8x8x16 c64）"
run 1800 python "$MAIN" verify --lattice 8 8 8 16 --prec c64

step "Step 2: 干净测量 + 资源统计（8x8x8x16，pairs 3）"
run 1800 python "$MAIN" clean --lattice 8 8 8 16 --prec c64 --levels 2 \
    --restart 10 --ct 1e5 --cmi 15 --pairs 3

step "Step 3: 批量基准（local 3 小格子，预算自动跳过超限）"
run 3600 python "$MAIN" bench --mode local --vram 16

step "Step 4: 参数扫描（8x8x8x16，9 配置，每配置 timeout 1800s）"
run 3600 python "$MAIN" sweep --lattice 8 8 8 16 --pairs 2 --timeout 1800

step "Step 5: 预算表（16G 档参考）"
run 120 python "$MAIN" budget --mode server --vram 16

step "Step 6: 汇总 + 表 + 图"
run 300 python "$MAIN" collect
run 300 python "$MAIN" mktable --mode server --vram 16
run 300 python "$MAIN" plots --vram 16
run 300 python "$MAIN" plots1 --file "$WORK/test12_sweep.json"

step "Step 7: 归档收敛日志（C++ 写死 REPO/logs/clover_multigrid.log）"
cp -f "$REPO/logs/clover_multigrid.log" "$WORK/clover_multigrid.log" 2>/dev/null \
  && echo "archived → $WORK/clover_multigrid.log" || echo "[warn] 无收敛日志"

echo
echo "=== 完成。完整输出: $LOG_FILE ==="
echo "产物目录: $WORK （test12_*.json / test12_*.png / test12_tbl_*.tex）"
