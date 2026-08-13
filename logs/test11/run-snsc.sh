#!/usr/bin/env bash
# test11 —— 服务器（SNSC）运行脚本。默认显存档 16GB；预留 32GB 档（VRAM=32）。
#
# 流程（dev74_1 服务器协议 + 16GB 档调整）：
#   Step 0  自检（nvidia-smi 显存 / torch / qcu 桥）
#   Step 1  强制闸门：8x8x8x16 干净测量 speedup_min >= GATE（默认 1.5）→ 不过即停
#   Step 2  参数扫描：8x16x16x16（9 配置）→ test11_sweep.json
#   Step 3  大格子全流程：16G 档 8x32x32x32（cold+warm 单卡可行）/
#                       32G 档 16x32x32x32（VRAM=32 启用）
#   Step 4  大格子 warm 探索：16G 档 16x32x32x32（cold 26.5GB 超档，需外部缓存，
#           OOM 仅记录不中断）/ 32G 档 16x32x32x64
#   Step 5  汇总断言（sweep + Step1 数据，gate）
#   Step 6  collect / mktable / plots / plots1
#   Step 7  归档收敛日志
#
# 特点：完整终端输出（tee 归档 run-snsc-<ts>.log）；每步 timeout 防卡壳；
#   gate 失败立即退出，其余步骤失败仅记录继续。
#
# 用法：
#   bash logs/test11/run-snsc.sh              # 实际执行（16GB 档）
#   VRAM=32 bash logs/test11/run-snsc.sh      # 预留 32GB 档（暂不启用）
#   GATE=1.5 可选；bash logs/test11/run-snsc.sh --dry-run 只打印
set -uo pipefail

REPO="${HOME}/PyQCU"
WORK="$REPO/logs/test11"
MAIN="$WORK/main.py"
TS=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$WORK/run-snsc-$TS.log"
DRY="${DRY:-${1:-}}"
[ "$DRY" = "--dry-run" ] && DRY=1 || DRY=0
VRAM="${VRAM:-16}"                 # 默认 16GB；预留 32GB 档
GATE="${GATE:-1.5}"

step() { echo; echo "===== $1 ====="; }
run() { # $1=timeout_s  "$@"=command ；返回 rc
  local tmo="$1"; shift
  if [ "$DRY" = "1" ]; then echo "[dry-run] timeout $tmo $*"; return 0; fi
  echo ">>> timeout $tmo $*"
  timeout "$tmo" "$@"
  return $?
}

cd "$REPO" || exit 1
source ./env.sh >/dev/null 2>&1
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== test11 server runner start $(date) | VRAM=${VRAM}G GATE=${GATE} ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"

step "Step 0: 环境自检"
run 120 nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
run 120 python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1e6, 'MB')"
run 120 python -c "from pyqcu.cuda import qcu; print('qcu bridge OK')"

# ---------- Step 1: 强制闸门 ----------
step "Step 1: 强制闸门 —— 8x8x8x16 c64 2L 干净测量（预期 speedup ~2.4x >= ${GATE}）"
run 5400 python "$MAIN" clean --lattice 8 8 8 16 --prec c64 --levels 2 \
    --restart 10 --ct 1e5 --cmi 15 --pairs 3
if [ "$DRY" = "1" ]; then
  echo "[dry-run] Step 1 通过检查逻辑略"
elif [ -n "$(ls -t "$WORK"/test11_clean_L8x8x8x16*.json 2>/dev/null | head -1)" ]; then
  GATE_FILE=$(ls -t "$WORK"/test11_clean_L8x8x8x16*.json | head -1)
  if timeout 120 python "$MAIN" check --gate "$GATE" --label "Step1 gate" --file "$GATE_FILE"; then
    echo "Step 1 通过：speedup >= ${GATE}，继续"
  else
    echo "Step 1 失败：speedup < ${GATE} 或无数据，停止。请检查 GPU 占用/时钟节流/日志。" >&2
    exit 1
  fi
else
  echo "Step 1 失败：无测量 json。停止。" >&2
  exit 1
fi

# ---------- Step 2: 参数扫描 ----------
step "Step 2: 参数优化扫描 —— 8x16x16x16（9 配置，每配置 timeout 1800s）"
run 7200 python "$MAIN" sweep --lattice 8 16 16 16 --pairs 3 --timeout 1800

# ---------- Step 3: 大格子全流程 ----------
if [ "$VRAM" -ge 32 ]; then
  step "Step 3: 大格子 16x32x32x32（32G 档：cold 26.5G / warm 13.5G，全流程可行）"
  run 28800 python "$MAIN" clean --lattice 16 32 32 32 --prec c64 --levels 2 \
      --restart 10 --ct 1e5 --cmi 15 --pairs 3
else
  step "Step 3: 大格子 8x32x32x32（16G 档：cold 13.3G / warm 6.8G，全流程可行）"
  run 28800 python "$MAIN" clean --lattice 8 32 32 32 --prec c64 --levels 2 \
      --restart 10 --ct 1e5 --cmi 15 --pairs 3
fi

# ---------- Step 4: 大格子 warm 探索（OOM 容忍，失败仅记录）----------
if [ "$VRAM" -ge 32 ]; then
  step "Step 4: 大格子 16x32x32x64 warm 探索（cold 53G 超档需分阶段，warm 27G 可行）"
  run 28800 python "$MAIN" clean --lattice 16 32 32 64 --prec c64 --levels 2 \
      --restart 10 --ct 1e5 --cmi 15 --pairs 2 || \
    echo "[warn] 16x32x32x64 失败（预期 cold OOM），已记录，继续。可先在其他 32G 卡构建 nullvec 缓存再 warm 复测"
else
  step "Step 4: 大格子 16x32x32x32 warm 探索（cold 26.5G 超 16G 档；若已有 nullvec 缓存则 warm 13.5G 可行）"
  run 28800 python "$MAIN" clean --lattice 16 32 32 32 --prec c64 --levels 2 \
      --restart 10 --ct 1e5 --cmi 15 --pairs 2 || \
    echo "[warn] 16x32x32x32 cold OOM（16G 档预期），已记录。方案：32G 卡构建缓存 → 同步 nullvec_cache → warm 复测"
fi

# ---------- Step 5: 汇总断言 ----------
step "Step 5: 汇总断言（sweep 数据，gate=${GATE}）"
if [ "$DRY" = "1" ]; then
  echo "[dry-run] check --gate ${GATE} --file $WORK/test11_sweep.json"
else
  timeout 120 python "$MAIN" check --gate "$GATE" --label "test11 final" \
      --file "$WORK/test11_sweep.json" || echo "[warn] 部分配置未达标（exit $?），见上表"
fi

# ---------- Step 6: 汇总 + 表 + 图 ----------
step "Step 6: collect / budget / mktable / plots / plots1"
run 300 python "$MAIN" collect
run 300 python "$MAIN" budget --mode server --vram "$VRAM"
run 300 python "$MAIN" mktable --mode server --vram "$VRAM"
run 300 python "$MAIN" plots --vram "$VRAM"
run 300 python "$MAIN" plots1 --file "$WORK/test11_sweep.json"

# ---------- Step 7: 归档 ----------
step "Step 7: 归档收敛日志"
cp -f "$REPO/logs/clover_multigrid.log" "$WORK/clover_multigrid.log" 2>/dev/null \
  && echo "archived → $WORK/clover_multigrid.log" || echo "[warn] 无收敛日志"

echo
echo "=== 完成。完整输出: $LOG_FILE ==="
echo "产物目录: $WORK （test11_*.json / test11_*.png / test11_tbl_*.tex）"
echo "说明：nullvec 缓存共享 $REPO/logs/nullvec_cache（PYQCU_NULLVEC_CACHE 可覆盖）；"
echo "      首次构建粗算子耗时长（8x32x32x32 ~1-2h），缓存命中后秒级。"
