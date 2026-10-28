#!/usr/bin/env bash
# Four independent warmed MPI groups in A/B/B/A order for CUDA Legacy-MG.
#
# Required:
#   PAIR_BASELINE_LIB   instrumented baseline libqcu.so (scalar-route build)
#   PAIR_OPTIMIZED_LIB  optimized libqcu.so (pair-route build)
#   PAIR_OUTPUT_ROOT    directory for group reports and evidence
#
# Optional:
#   PAIR_RANKS=2, PAIR_GRID='2 1 1 1', PAIR_CASE_DIR=<prepared case>
#   PAIR_COARSE_CACHE=<h5>, PAIR_WARMUPS=1, PAIR_SAMPLES=5
#   PAIR_GROUP_TIMEOUT=120, PAIR_TOTAL_TIMEOUT=540, PAIR_DRY_RUN=0
set -euo pipefail

trap 'rc=$?; printf "ERROR line=%s rc=%s command=%q\n" "$LINENO" "$rc" "$BASH_COMMAND" >&2' ERR

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(CDPATH= cd -- "${script_dir}/../../../.." && pwd)
sampler="${script_dir}/cuda_pair_reduce_samples.py"
python_bin=${PYQCU_PYTHON:-python3}
mpirun_bin=${PYQCU_MPIRUN:-mpirun}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

baseline_lib=${PAIR_BASELINE_LIB:-}
optimized_lib=${PAIR_OPTIMIZED_LIB:-}
output_root=${PAIR_OUTPUT_ROOT:-}
ranks=${PAIR_RANKS:-2}
warmups=${PAIR_WARMUPS:-1}
samples=${PAIR_SAMPLES:-5}
group_timeout=${PAIR_GROUP_TIMEOUT:-120}
total_timeout=${PAIR_TOTAL_TIMEOUT:-540}
dry_run=${PAIR_DRY_RUN:-0}

[[ -n ${baseline_lib} ]] || die 'PAIR_BASELINE_LIB is required'
[[ -n ${optimized_lib} ]] || die 'PAIR_OPTIMIZED_LIB is required'
[[ -n ${output_root} ]] || die 'PAIR_OUTPUT_ROOT is required'
[[ -f ${baseline_lib} ]] || die "baseline library not found: ${baseline_lib}"
[[ -f ${optimized_lib} ]] || die "optimized library not found: ${optimized_lib}"
[[ -f ${sampler} ]] || die "sampler not found: ${sampler}"
[[ ${warmups} =~ ^[0-9]+$ ]] && ((warmups >= 1)) || \
  die 'PAIR_WARMUPS must be at least 1'
[[ ${samples} =~ ^[0-9]+$ ]] && ((samples >= 5)) || \
  die 'PAIR_SAMPLES must be at least 5'
[[ ${group_timeout} =~ ^[0-9]+$ ]] && ((group_timeout > 0)) || \
  die 'PAIR_GROUP_TIMEOUT must be positive'
[[ ${total_timeout} =~ ^[0-9]+$ ]] && ((total_timeout > 0 && total_timeout <= 540)) || \
  die 'PAIR_TOTAL_TIMEOUT must be in [1, 540] seconds'

requested_order=${PAIR_ORDER:-'A B B A'}
[[ ${requested_order} == 'A B B A' ]] || \
  die "PAIR_ORDER is fixed to 'A B B A'"
order=(A B B A)

case_dir=${PAIR_CASE_DIR:-${repo_root}/logs/qcu_backend_sync_20261028/cuda_smoke2/case}
gauge=${case_dir}/gauge_global_c64.h5
source_path=${case_dir}/source_global_c64.h5
coarse_cache=${PAIR_COARSE_CACHE:-${case_dir}/coarse_c64_lv2_E12_6_mg2x2x2x1_nvi1_s42.h5}
[[ -f ${gauge} ]] || die "gauge not found: ${gauge}"
[[ -f ${source_path} ]] || die "source not found: ${source_path}"
[[ -f ${coarse_cache} ]] || die "coarse cache not found: ${coarse_cache}"

case "${ranks}" in
  1) die 'PAIR_RANKS=1 has no inter-rank MPI reduction and cannot prove this optimization' ;;
  2) default_grid='2 1 1 1' ;;
  4) default_grid='2 2 1 1' ;;
  *) die 'PAIR_RANKS must be 1, 2, or 4' ;;
esac
grid_spec=${PAIR_GRID:-${default_grid}}
read -r -a grid_args <<< "${grid_spec}"
((${#grid_args[@]} == 4)) || die "PAIR_GRID must contain four integers: ${grid_spec}"

baseline_lib=$(readlink -f "${baseline_lib}")
optimized_lib=$(readlink -f "${optimized_lib}")
baseline_sha=$(sha256sum "${baseline_lib}" | cut -d' ' -f1)
optimized_sha=$(sha256sum "${optimized_lib}" | cut -d' ' -f1)
[[ ${baseline_sha} != "${optimized_sha}" ]] || \
  die 'baseline and optimized libraries have identical SHA256'

mkdir -p "${output_root}/preflight"
printf 'baseline=%s\noptimized=%s\n' "${baseline_lib}" "${optimized_lib}" \
  > "${output_root}/libraries.txt"
printf '%s  %s\n%s  %s\n' \
  "${baseline_sha}" "${baseline_lib}" \
  "${optimized_sha}" "${optimized_lib}" >> "${output_root}/libraries.txt"
{
  printf 'host=%s\n' "$(hostname)"
  printf 'job_id=%s\n' "${SLURM_JOB_ID:-}"
  printf 'partition=%s\n' "${SLURM_JOB_PARTITION:-}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
  printf 'rocr_visible_devices=%s\n' "${ROCR_VISIBLE_DEVICES:-}"
  printf 'date=%s\n' "$(date -Is)"
} > "${output_root}/environment.txt"

mapfile -t extensions < <(
  find "${repo_root}/pyqcu/cuda" -maxdepth 1 -type f -name 'qcu*.so' -print)
((${#extensions[@]} == 1)) || \
  die "expected one Cython qcu extension, found ${#extensions[@]}"
extension=${extensions[0]}

resolve_ldd_library() {
  local label=$1
  local library=$2
  local library_dir report loaded
  library_dir=$(dirname -- "${library}")
  report="${output_root}/preflight/${label}-ldd.txt"
  LD_LIBRARY_PATH="${library_dir}:${LD_LIBRARY_PATH:-}" \
    ldd "${extension}" >"${report}" 2>&1 || true
  loaded=$(awk '$1 == "libqcu.so" {print $3; exit}' "${report}")
  [[ -n ${loaded} ]] || die "cannot resolve libqcu.so for ${label}"
  loaded=$(readlink -f "${loaded}")
  [[ ${loaded} == "${library}" ]] || \
    die "${label} resolved ${loaded}, expected ${library}"
  printf '%s\n' "${loaded}"
}

if [[ ${dry_run} != 1 ]]; then
  baseline_resolved=$(resolve_ldd_library baseline "${baseline_lib}")
  optimized_resolved=$(resolve_ldd_library optimized "${optimized_lib}")
else
  baseline_resolved=${baseline_lib}
  optimized_resolved=${optimized_lib}
fi

capture_command() {
  local output=$1
  shift
  local rc
  set +e
  timeout --signal=TERM --kill-after=5s 5s "$@" >"${output}" 2>"${output}.err"
  rc=$?
  set -e
  printf '%s\n' "${rc}" >"${output}.rc"
}

capture_device_metadata() {
  local destination=$1
  mkdir -p "${destination}"
  if command -v rocm-smi >/dev/null 2>&1; then
    capture_command "${destination}/rocm-devices.json" \
      rocm-smi --showuniqueid --showclocks --showtemp --showpower \
      --showproductname --json
  else
    printf 'rocm-smi unavailable\n' >"${destination}/rocm-unavailable.txt"
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    capture_command "${destination}/nvidia-devices.csv" \
      nvidia-smi \
      --query-gpu=index,uuid,name,clocks.current.graphics,clocks.current.memory,temperature.gpu,power.draw,power.limit \
      --format=csv,noheader
  else
    printf 'nvidia-smi unavailable\n' >"${destination}/nvidia-unavailable.txt"
  fi
}

if [[ ${SLURM_JOB_PARTITION:-} == kshdnormal ]]; then
  visible=${CUDA_VISIBLE_DEVICES:-}
  [[ -n ${visible} ]] || die 'DCU allocation has no CUDA_VISIBLE_DEVICES'
  IFS=',' read -r -a visible_devices <<< "${visible}"
  ((${#visible_devices[@]} >= ranks)) || die \
    "allocation exposes ${#visible_devices[@]} device(s), fewer than ${ranks} ranks"
fi

if [[ ${dry_run} != 1 ]]; then
  capture_device_metadata "${output_root}/preflight/devices"
fi

deadline=$((SECONDS + total_timeout))
audit_file="${output_root}/five-minute-audit.txt"

audit_if_due() {
  local elapsed=$((SECONDS - (deadline - total_timeout)))
  if ((elapsed < 300)) || [[ -s ${audit_file} ]]; then
    return 0
  fi
  {
    printf 'timestamp=%s\n' "$(date -Is)"
    printf 'job_id=%s\n' "${SLURM_JOB_ID:-}"
    printf 'elapsed_seconds=%s\n' "${elapsed}"
    printf 'completed_groups:\n'
    local report
    for report in "${output_root}"/*/group.json; do
      [[ -f ${report} ]] || continue
      printf '  %s ' "${report}"
      grep -o '"status"[[:space:]]*:[[:space:]]*"[^"]*"' "${report}" \
        | head -1 || true
    done
    printf 'device_state:\n'
    if command -v rocm-smi >/dev/null 2>&1; then
      timeout --signal=TERM --kill-after=2s 5s \
        rocm-smi --showuniqueid --showuse --showmemuse --json 2>&1 || true
    else
      printf '  rocm-smi unavailable\n'
    fi
    printf 'rank_processes:\n'
    pgrep -af "cuda_pair_reduce_samples.py" || true
  } > "${audit_file}"
}

for slot in "${!order[@]}"; do
  audit_if_due
  variant=${order[$slot]}
  case "${variant}" in
    A)
      lib=${baseline_lib}
      lib_sha=${baseline_sha}
      ldd_lib=${baseline_resolved}
      label="a-baseline-${slot}"
      ;;
    B)
      lib=${optimized_lib}
      lib_sha=${optimized_sha}
      ldd_lib=${optimized_resolved}
      label="b-optimized-${slot}"
      ;;
  esac
  lib_dir=$(dirname -- "${lib}")
  sample_dir="${output_root}/${label}"
  mkdir -p "${sample_dir}"
  monitor="${sample_dir}/monitor.txt"
  group_stdout="${sample_dir}/stdout.txt"
  group_stderr="${sample_dir}/stderr.txt"

  command=(
    "${python_bin}" "${sampler}" run
    --variant "${variant}"
    --label "${label}"
    --slot "${slot}"
    --output-root "${sample_dir}"
    --gauge "${gauge}"
    --source "${source_path}"
    --coarse-cache "${coarse_cache}"
    --lat 8 8 8 16
    --grid "${grid_args[@]}"
    --mass -0.277
    --csw 1.160920226
    --atol 1e-6
    --max-iter 200
    --dtype c64
    --levels 2
    --dof 12 6
    --mg-grid 2 2 2 1
    --restart 5
    --coarse-max-iter 200
    --coarse-tol-factor 1e4
    --nv-iters 1
    --seed 42
    --warmups "${warmups}"
    --samples "${samples}"
    --library "${lib}"
    --ldd-library "${ldd_lib}"
    --library-sha256 "${lib_sha}"
    --device-metadata-dir "${sample_dir}/device_metadata"
  )

  if [[ ${dry_run} == 1 ]]; then
    printf '%s: ' "${label}"
    printf '%q ' env "LD_LIBRARY_PATH=${lib_dir}:${LD_LIBRARY_PATH:-}" \
      "PYTHONPATH=${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
      OMP_NUM_THREADS=1 "${mpirun_bin}" --oversubscribe --bind-to none \
      -np "${ranks}" "${command[@]}"
    printf '\n'
    continue
  fi

  capture_device_metadata "${sample_dir}/device_metadata/before"
  printf '%s START label=%s variant=%s slot=%s ranks=%s grid=%s lib=%s\n' \
    "$(date -Is)" "${label}" "${variant}" "${slot}" "${ranks}" \
    "${grid_spec}" "${lib}" | tee "${monitor}"
  remaining=$((deadline - SECONDS))
  ((remaining > 0)) || die "total ABBA deadline exhausted before ${label}"
  run_timeout=${group_timeout}
  ((run_timeout > remaining)) && run_timeout=${remaining}
  started=$(date +%s)
  set +e
  timeout --signal=TERM --kill-after=10s "${run_timeout}s" \
    env "LD_LIBRARY_PATH=${lib_dir}:${LD_LIBRARY_PATH:-}" \
      "PYTHONPATH=${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
      OMP_NUM_THREADS=1 "QCU_LOG_DIR=${sample_dir}/native-logs" \
      "${mpirun_bin}" --oversubscribe --bind-to none \
      -np "${ranks}" "${command[@]}" >"${group_stdout}" 2>"${group_stderr}"
  rc=$?
  set -e
  elapsed=$(($(date +%s) - started))
  capture_device_metadata "${sample_dir}/device_metadata/after"
  printf '%s END label=%s rc=%s elapsed=%ss group=%s\n' \
    "$(date -Is)" "${label}" "${rc}" "${elapsed}" \
    "${sample_dir}/group.json" | tee -a "${monitor}"
  if ((rc != 0)); then
    exit "${rc}"
  fi
  [[ -s ${sample_dir}/group.json ]] || die "${label} omitted group.json"
done

audit_if_due

if [[ ${dry_run} == 1 ]]; then
  printf 'ABBA DRY RUN ranks=%s grid=%s warmups=%s samples=%s\n' \
    "${ranks}" "${grid_spec}" "${warmups}" "${samples}"
  exit 0
fi

"${python_bin}" "${sampler}" aggregate \
  --input-root "${output_root}" \
  --output "${output_root}/aggregate.json"
printf 'ABBA COMPLETE ranks=%s grid=%s order=%s samples=%s\n' \
  "${ranks}" "${grid_spec}" "${order[*]}" "$((4 * samples))" \
  | tee "${output_root}/status.txt"
