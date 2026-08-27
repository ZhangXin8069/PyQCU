#!/usr/bin/env bash
# pyquda 对比套件一键运行（双进程隔离，dev87 F2）。
# 用法: bash examples/pyquda/run_all.sh [lat串 如 8x8x8x16] [tol]
set -e
LAT=${1:-8x8x8x16}
TOL=${2:-1e-8}
IFS=x read -ra L <<< "$LAT"
DIR=$(cd "$(dirname "$0")" && pwd)
cd "$DIR"
source ../../env.sh >/dev/null 2>&1

echo "=== 1/3 pyqcu 阶段（进程 A）==="
python run_pyqcu.py --lat "${L[@]}" --tol "$TOL" || exit 1
echo "=== 2/3 pyquda 阶段（进程 B，隔离）==="
python run_pyquda.py --lat "${L[@]}" --tol "$TOL" || exit 1
echo "=== 3/3 聚合对比 + 作图 ==="
python compare.py --lat "${L[@]}" || exit 1
echo "DONE: examples/pyquda/out/compare_${LAT}.md + .png"