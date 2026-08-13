#!/usr/bin/env bash
# test12 —— 2026-08-14 本地一键脚本（RTX 4060 8GB）：
# 实测「服务器确认加速比 >1.2」的格子大小（dev73_5 V100-32G 实测基准）：
#   * 8x8x8x16   c64 2L r10 ct1e5 cmi15  服务器 speedup = 2.43x
#   * 8x8x8x16   c64 3L r10 ct1e5 cmi15  服务器 3L>2L 相对行为（未单列）
#   * 8x16x16x16 c64 3L r10 ct1e5 cmi15  服务器 speedup = 1.32x
# 本地（4060）实测预期 speedup < 1（MG 单迭代成本高是硬件特性，AGENTS.md
# 有记载）；本脚本作用 = 在本地实测这些服务器级配置，输出「本地实测 vs
# 服务器参考」对照表，供跨环境比对，不以本地结果推断服务器。
#
# 用法：
#   bash logs/test12/run-local-v20260814.sh            # 实际执行
#   bash logs/test12/run-local-v20260814.sh --dry-run  # 只打印命令
set -uo pipefail

REPO="${HOME}/PyQCU"
WORK="$REPO/logs/test12"
MAIN="$WORK/main.py"
TS=$(date +%Y%m%d-%H%M%S)

# ---- 版本目录：v<YYYYMMDDHHMM>；同分钟重跑加 -<SS> 防覆盖 ----
VDIR="$WORK/v$(date +%Y%m%d%H%M)"
if [ -e "$VDIR" ]; then VDIR="$VDIR-$(date +%S)"; fi
mkdir -p "$VDIR"
export TEST12_OUTDIR="$VDIR"
LOG_FILE="$VDIR/run-local-v20260814-$TS.log"

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
echo "=== test12 local runner (v20260814) start $(date) ==="
echo "版本目录: $VDIR"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

step "Step 0: 环境自检"
run 120 python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
run 120 python -c "from pyqcu.cuda import qcu; print('qcu bridge OK')"

step "Step 1: 正确性验证（8x8x8x16 c64）"
run 1800 python "$MAIN" verify --lattice 8 8 8 16 --prec c64

step "Step 2: 服务器 >1.2 配置① —— 8x8x8x16 2L r10（V100-32G 实测 2.43x；本地 nullvec 缓存命中）"
run 1800 python "$MAIN" clean --lattice 8 8 8 16 --prec c64 --levels 2 \
    --restart 10 --ct 1e5 --cmi 15 --pairs 3

step "Step 3: 服务器 >1.2 配置② —— 8x8x8x16 3L r10（3L>2L 相对行为跨 GPU 一致）"
run 1800 python "$MAIN" clean --lattice 8 8 8 16 --prec c64 --levels 3 \
    --restart 10 --ct 1e5 --cmi 15 --pairs 3

step "Step 4: 服务器 >1.2 配置③ —— 8x16x16x16 3L r10（V100-32G 实测 1.32x；lv2 粗算子首次构建可能较久）"
run 7200 python "$MAIN" clean --lattice 8 16 16 16 --prec c64 --levels 3 \
    --restart 10 --ct 1e5 --cmi 15 --pairs 3

step "Step 5: 本地实测 vs 服务器参考对照"
run 300 python "$MAIN" collect
run 300 python - "$VDIR" << 'PYEOF'
import json, sys
vd = sys.argv[1]
ref = {
    ((8, 8, 8, 16), 2, 10): ("2.43x", "dev73_5 V100-32G"),
    ((8, 8, 8, 16), 3, 10): ("3L>2L（服务器 3L 未单列）", "dev73_5"),
    ((8, 16, 16, 16), 3, 10): ("1.32x", "dev73_5 V100-32G"),
}
try:
    data = json.load(open(vd + "/test12_results.json"))
except Exception as e:
    print("[warn] 无 test12_results.json（collect 未生成）:", e); sys.exit(0)
print(f"{'配置':<38}{'本地 speedup':>12}  {'服务器参考':>28}")
print("-" * 84)
for r in data["results"]:
    key = (tuple(r["lattice"]), r["levels"], r["restart"])
    srv = ref.get(key)
    if srv is None:
        continue
    print(f"{r['label']:<38}{r['speedup_min']:>11.3f}x  {srv[0]:>14}（{srv[1]}）")
print()
print("注：本地 4060 上 MG 恒慢（speedup<1）是硬件特性；")
print("    加速比以服务器为准，本地结果仅用于功能验证与跨环境对照。")
PYEOF

step "Step 6: 归档收敛日志（C++ 写死 REPO/logs/clover_multigrid.log）"
cp -f "$REPO/logs/clover_multigrid.log" "$VDIR/clover_multigrid.log" 2>/dev/null \
  && echo "archived → $VDIR/clover_multigrid.log" || echo "[warn] 无收敛日志"

echo
echo "=== 完成。完整输出: $LOG_FILE ==="
echo "版本目录: $VDIR（test12_clean_L*.json + test12_results.json + 对照表 + 收敛日志）"
