#!/usr/bin/env bash
# Verify all timed CUDA pair-reduction samples with an independent CPU solve.
#
# This script must run outside the DCU partition.  It reads the group reports
# written by cuda_pair_reduce_abba.sh and checks every timed solution.
set -euo pipefail

trap 'rc=$?; printf "ERROR line=%s rc=%s command=%q\n" "$LINENO" "$rc" "$BASH_COMMAND" >&2' ERR

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(CDPATH= cd -- "${script_dir}/../../../.." && pwd)
sampler="${script_dir}/cuda_pair_reduce_samples.py"
residual="${script_dir}/quda_residual.py"
python_bin=${PYQCU_PYTHON:-python3}
mpirun_bin=${PYQCU_MPIRUN:-mpirun}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

input_root=${1:-}
output_root=${2:-}
ranks=${3:-2}
grid_spec=${4:-'2 1 1 1'}
if (($# > 4)); then
  shift 4
  grid_spec="$*"
fi
sample_timeout=${PAIR_RESIDUAL_TIMEOUT:-180}
case_dir=${PAIR_CASE_DIR:-${repo_root}/logs/qcu_backend_sync_20261028/cuda_smoke2/case}
gauge=${PAIR_GAUGE:-${case_dir}/gauge_global_c64.h5}
source_path=${PAIR_SOURCE:-${case_dir}/source_global_c64.h5}

[[ -n ${input_root} ]] || \
  die 'usage: verify_cuda_pair_reduce.sh INPUT_ROOT OUTPUT_ROOT [RANKS GRID]'
[[ -n ${output_root} ]] || die 'OUTPUT_ROOT is required'
[[ -d ${input_root} ]] || die "input root not found: ${input_root}"
[[ -f ${sampler} ]] || die "sampler not found: ${sampler}"
[[ -f ${residual} ]] || die "residual verifier not found: ${residual}"
[[ -f ${gauge} ]] || die "gauge not found: ${gauge}"
[[ -f ${source_path} ]] || die "source not found: ${source_path}"
[[ ${SLURM_JOB_PARTITION:-} != kshdnormal ]] || \
  die 'refusing CPU-only residual verification on the kshdnormal DCU partition'
read -r -a grid <<< "${grid_spec}"
((${#grid[@]} == 4)) || die "grid must contain four integers: ${grid_spec}"

mapfile -t entries < <(
  PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" - "${input_root}" "${sampler}" <<'PY'
import sys
from pathlib import Path

from pyqcu.testing.simple.mpi.cuda_pair_reduce_samples import (
    load_group_reports,
)

root = Path(sys.argv[1])
reports = load_group_reports(root)
if not reports:
    raise SystemExit("no group reports found")
for report in reports:
    for sample in report.get("samples", []):
        print(
            report["label"],
            int(sample["index"]),
            sample["solution"],
            sep="\t",
        )
PY
)
((${#entries[@]} > 0)) || die "no timed samples found under ${input_root}"

mkdir -p "${output_root}"
started_all=$(date +%s)
for entry in "${entries[@]}"; do
  IFS=$'\t' read -r group_label sample_index solution <<< "${entry}"
  [[ -f ${solution} ]] || die "solution not found: ${solution}"
  sample_root="${output_root}/${group_label}/samples/sample-$(printf '%03d' "${sample_index}")"
  mkdir -p "${sample_root}"
  printf '%s START group=%s sample=%s\n' \
    "$(date -Is)" "${group_label}" "${sample_index}" \
    | tee "${sample_root}/monitor.txt"
  started=$(date +%s)
  set +e
  timeout --signal=TERM --kill-after=10s "${sample_timeout}s" \
    env OMP_NUM_THREADS=1 \
    "PYTHONPATH=${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${mpirun_bin}" --oversubscribe --bind-to none \
    -np "${ranks}" \
    "${python_bin}" "${residual}" \
    --gauge "${gauge}" \
    --source "${source_path}" \
    --solution "${solution}" \
    --lat 8 8 8 16 \
    --grid "${grid[@]}" \
    --ranks "${ranks}" \
    --dtype c64 \
    --mass -0.277 \
    --csw 1.160920226 \
    --atol 1e-6 \
    --report "${sample_root}/residual.json" \
    >"${sample_root}/stdout.txt" 2>"${sample_root}/stderr.txt"
  rc=$?
  set -e
  elapsed=$(($(date +%s) - started))
  printf '%s END group=%s sample=%s rc=%s elapsed=%ss\n' \
    "$(date -Is)" "${group_label}" "${sample_index}" "${rc}" "${elapsed}" \
    | tee -a "${sample_root}/monitor.txt"
  if ((rc != 0)); then
    exit "${rc}"
  fi
  [[ -s ${sample_root}/residual.json ]] || \
    die "residual verifier did not publish ${sample_root}/residual.json"
done

"${python_bin}" "${sampler}" verify-residuals \
  --input-root "${input_root}" \
  --residual-root "${output_root}" \
  --output "${output_root}/residual-summary.json"
elapsed_all=$(($(date +%s) - started_all))
printf 'RESIDUAL COMPLETE ranks=%s grid=%s samples=%s elapsed=%ss\n' \
  "${ranks}" "${grid_spec}" "${#entries[@]}" "${elapsed_all}" \
  | tee "${output_root}/status.txt"
