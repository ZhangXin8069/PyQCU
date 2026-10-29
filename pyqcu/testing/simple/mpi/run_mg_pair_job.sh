#!/usr/bin/env bash
set -euo pipefail

# Run same-node CUDA-QCU and QUDA Legacy-MG jobs in ABBA order.
# This script is intended to run inside one Slurm allocation.  The Python
# entry points launch their own mpirun, so this script itself must be a single
# process.

repo_root=${PYQCU_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}
site_env=${PYQCU_SLURM_ENV:-${BSCC_ENV:-}}
if [[ -n ${site_env} ]]; then
  # shellcheck disable=SC1090
  source "${site_env}"
fi
if [[ -f ${repo_root}/env.sh ]]; then
  # shellcheck disable=SC1091
  source "${repo_root}/env.sh"
fi

python_bin=${PYQCU_PYTHON:-python3}
lime=${PYQCU_LIME:?PYQCU_LIME is required}
source_file=${PYQCU_SOURCE:?PYQCU_SOURCE is required}
qcu_cache=${PYQCU_COARSE_CACHE:?PYQCU_COARSE_CACHE is required}
qcu_cache_levels=${PYQCU_COARSE_CACHE_LEVELS:-}
quda_install=${QUDA_INSTALL:-${HOME}/lib/quda-develop/build}
pyquda_root=${PYQCU_PYQUDA_ROOT:-${repo_root}/refer/git-rep/PyQUDA}
result_root=${PYQCU_RESULT_ROOT:-${repo_root}/data/qcu_mg_pair/$(date +%Y%m%d%H%M%S)}
ranks=${PYQCU_RANKS:-1}
grid=${PYQCU_GRID:-1,1,1,1}
lattice=${PYQCU_LATTICE:-24,24,24,64}
mass=${PYQCU_MASS:--0.277}
csw=${PYQCU_CSW:-1.160920226}
atol=${PYQCU_ATOL:-1e-6}
max_iter=${PYQCU_MAX_ITER:-1000}
levels=${PYQCU_LEVELS:-2}
min_size=${PYQCU_MIN_SIZE:-4}
dof=${PYQCU_DOF:-12,12}
mg_grid=${PYQCU_MG_GRID:-2,2,2,2}
nvec=${PYQCU_QUDA_NVEC:-12}
mg_block=${PYQCU_QUDA_MG_BLOCK:-2,2,2,2}
quda_nu_pre=${PYQCU_QUDA_NU_PRE:-}
quda_nu_post=${PYQCU_QUDA_NU_POST:-}
quda_smoother=${PYQCU_QUDA_SMOOTHER:-}
quda_coarse_max_iter=${PYQCU_QUDA_COARSE_MAX_ITER:-}
restart=${PYQCU_RESTART:-5}
coarse_max_iter=${PYQCU_COARSE_MAX_ITER:-200}
coarse_tol_factor=${PYQCU_COARSE_TOL_FACTOR:-1e4}
nv_iters=${PYQCU_NV_ITERS:-1}
warmups=${PYQCU_WARMUPS:-1}
repeats=${PYQCU_REPEATS:-5}
seed=${PYQCU_SEED:-42}
wall_limit=${PYQCU_WALL_LIMIT:-570}
child_timeout=${PYQCU_CHILD_TIMEOUT:-540}
mpirun=${PYQCU_MPIRUN:-mpirun}
cycles=${PYQCU_MG_ABBA_CYCLES:-2}

case ${cycles} in
  1) sequence=(qcu quda) ;;
  2) sequence=(qcu quda quda qcu) ;;
  *) printf 'PYQCU_MG_ABBA_CYCLES must be 1 or 2\n' >&2; exit 2 ;;
esac
job_audited=0

if [[ ! -f ${lime} || ! -f ${source_file} || ! -f ${qcu_cache} ]]; then
  printf 'missing required input: lime=%s source=%s cache=%s\n' \
    "${lime}" "${source_file}" "${qcu_cache}" >&2
  exit 2
fi
if [[ ! -d ${quda_install} ]]; then
  printf 'missing QUDA install: %s\n' "${quda_install}" >&2
  exit 2
fi

"${python_bin}" "${repo_root}/pyqcu/testing/simple/mpi/mpi_ranks_power_of_two.py" \
  --quiet "${ranks}"
IFS=',' read -r gx gy gz gt <<< "${grid}"
if (( gx * gy * gz * gt != ranks )); then
  printf 'grid %s does not contain %s ranks\n' "${grid}" "${ranks}" >&2
  exit 2
fi
IFS=',' read -r lx ly lz lt <<< "${lattice}"
if (( lx % gx || ly % gy || lz % gz || lt % gt )); then
  printf 'grid %s does not divide lattice %s\n' "${grid}" "${lattice}" >&2
  exit 2
fi

mkdir -p "${result_root}"
result_root=$(cd "${result_root}" && pwd)
unset QUDA_MG_TRACE QUDA_MG_TRACE_FILE PYQCU_STRICT_TRACE_FILE
export QUDA_ENABLE_TUNING=0
export QUDA_ENABLE_P2P=0
export PYQCU_MPI_DEVICE_AWARE=0
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf 'schema_version=pyqcu.mg-pair.v1\nresult_root=%s\nranks=%s\ngrid=%s\nlattice=%s\nwarmups=%s\nrepeats=%s\n' \
  "${result_root}" "${ranks}" "${grid}" "${lattice}" "${warmups}" "${repeats}" \
  > "${result_root}/matrix.env"
{
  printf 'host=%s\n' "$(hostname)"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-unset}"
  printf 'slurm_job_gpus=%s\n' "${SLURM_JOB_GPUS:-unset}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  printf 'quda_install=%s\n' "${quda_install}"
  printf 'pyquda_root=%s\n' "${pyquda_root}"
  printf 'python=%s\n' "$(command -v "${python_bin}")"
  printf 'mpirun=%s\n' "$(command -v "${mpirun}")"
  printf 'lime=%s\nsource=%s\nqcu_cache=%s\n' \
    "${lime}" "${source_file}" "${qcu_cache}"
  "${python_bin}" --version 2>&1
  module list 2>&1 || true
  nvidia-smi -L 2>&1 || true
} > "${result_root}/environment.txt"

run_one() {
  local backend=$1
  local sequence_index=$2
  local case_dir="${result_root}/${sequence_index}-${backend}-r${ranks}"
  local stdout_path="${case_dir}/stdout.txt"
  local stderr_path="${case_dir}/stderr.txt"
  local report_path="${case_dir}/report.json"
  local solution_path="${case_dir}/solution.h5"
  local monitor_path="${case_dir}/monitor.txt"
  local command=()
  local -a env_args=()
  mkdir -p "${case_dir}"

  if [[ ${backend} == qcu ]]; then
    command=(
      "${python_bin}" -B
      "${repo_root}/pyqcu/testing/simple/mpi/qcu_multigrid.py"
      --gauge "${lime}" --source "${source_file}"
      --lat ${lattice//,/ } --grid ${grid//,/ } --ranks "${ranks}"
      --mass "${mass}" --csw "${csw}" --dtype c64
      --atol "${atol}" --max-iter "${max_iter}"
      --levels "${levels}" --min-size "${min_size}"
      --dof ${dof//,/ } --mg-grid ${mg_grid//,/ }
      --restart "${restart}" --coarse-max-iter "${coarse_max_iter}"
      --coarse-tol-factor "${coarse_tol_factor}" --nv-iters "${nv_iters}"
      --seed "${seed}" --warmup "${warmups}" --repeat "${repeats}"
      --coarse-cache "${qcu_cache}" --build-device cuda --libqcu default
      --output "${solution_path}" --report "${report_path}")
    if [[ -n ${qcu_cache_levels} ]]; then
      IFS=':' read -r -a cache_levels <<< "${qcu_cache_levels}"
      for cache_level in "${cache_levels[@]}"; do
        if [[ -z ${cache_level} || ! -f ${cache_level} ]]; then
          printf 'missing coarse-cache level: %s\n' "${cache_level}" >&2
          return 2
        fi
        command+=(--coarse-cache-level "${cache_level}")
      done
    fi
    env_args=(
      "PYTHONPATH=${repo_root}:${PYTHONPATH:-}"
      "PYQCU_LIBQCU=${repo_root}/cpp/cuda/qcu/libqcu.so"
      "LD_LIBRARY_PATH=${repo_root}/cpp/cuda/qcu:${LD_LIBRARY_PATH:-}")
  else
    command=(
      "${python_bin}" -B
      "${repo_root}/pyqcu/testing/simple/mpi/quda_multigrid.py"
      --gauge "${lime}" --source "${source_file}"
      --lat ${lattice//,/ } --grid ${grid//,/ } --ranks "${ranks}"
      --mass "${mass}" --csw "${csw}" --dtype c64
      --tol "${atol}" --max-iter "${max_iter}"
      --levels "${levels}" --nvec "${nvec}" --block ${mg_block//,/ }
      --warmups "${warmups}" --repeats "${repeats}" --seed "${seed}"
      --solution-output "${solution_path}" --report "${report_path}")
    env_args=(
      "QUDA_INSTALL=${quda_install}"
      "QUDA_BUILD_DIR=${quda_install}"
      "QUDA_PATH=${quda_install}"
      "PYQCU_QUDA_ARRAY_BACKEND=torch"
      "PYTHONPATH=${repo_root}:${pyquda_root}:${pyquda_root}/pyquda_core:${PYTHONPATH:-}"
      "LD_LIBRARY_PATH=${quda_install}/lib:${LD_LIBRARY_PATH:-}")
  fi
  if [[ ${backend} == quda ]]; then
    [[ -n ${quda_nu_pre} ]] && command+=(--nu-pre "${quda_nu_pre}")
    [[ -n ${quda_nu_post} ]] && command+=(--nu-post "${quda_nu_post}")
    [[ -n ${quda_smoother} ]] && command+=(--smoother "${quda_smoother}")
    [[ -n ${quda_coarse_max_iter} ]] && command+=(
      --coarse-solver-maxiter "${quda_coarse_max_iter}")
  fi

  printf 'START %s backend=%s sequence=%s\n' \
    "$(date --iso-8601=seconds)" "${backend}" "${sequence_index}" \
    > "${monitor_path}"
  local started=$SECONDS
  local audited=0
  local timed_out=0
  env "${env_args[@]}" setsid stdbuf -oL -eL "${command[@]}" \
    > "${stdout_path}" 2> "${stderr_path}" &
  local pid=$!
  while kill -0 "${pid}" 2>/dev/null; do
    local now=$SECONDS
    local child_elapsed=$((now - started))
    local job_elapsed=$((now - job_started))
    if (( child_elapsed >= 300 && audited == 0 )); then
      audited=1
      {
        printf 'AUDIT_5MIN %s elapsed=%ss\n' \
          "$(date --iso-8601=seconds)" "${child_elapsed}"
        ps -eo pid,ppid,stat,etime,%cpu,%mem,rss,cmd --forest |
          grep -E '(mpirun|python|nvidia-smi)' | grep -v grep || true
        nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw \
          --format=csv,noheader 2>&1 || true
        tail -100 "${stdout_path}" || true
        tail -100 "${stderr_path}" || true
      } >> "${monitor_path}"
    fi
    if (( job_elapsed >= 300 && job_audited == 0 )); then
      job_audited=1
      {
        printf 'MATRIX_AUDIT_5MIN %s elapsed=%ss\n' \
          "$(date --iso-8601=seconds)" "${job_elapsed}"
        ps -eo pid,ppid,stat,etime,%cpu,%mem,rss,cmd --forest |
          grep -E '(mpirun|python|nvidia-smi)' | grep -v grep || true
        nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,power.draw \
          --format=csv,noheader 2>&1 || true
        tail -100 "${stdout_path}" || true
        tail -100 "${stderr_path}" || true
      } >> "${monitor_path}"
    fi
    local terminate_reason=""
    if (( child_elapsed >= child_timeout )); then
      terminate_reason="child_timeout"
    elif (( job_elapsed >= wall_limit )); then
      terminate_reason="matrix_wall_limit"
    fi
    if [[ -n ${terminate_reason} ]]; then
      timed_out=1
      {
        printf 'AUDIT_TERMINATE %s reason=%s child_elapsed=%ss job_elapsed=%ss child_limit=%ss job_limit=%ss\n' \
          "$(date --iso-8601=seconds)" "${terminate_reason}" \
          "${child_elapsed}" "${job_elapsed}" "${child_timeout}" "${wall_limit}"
        ps -eo pid,ppid,stat,etime,%cpu,%mem,rss,cmd --forest |
          grep -E '(mpirun|python|nvidia-smi)' | grep -v grep || true
        tail -100 "${stdout_path}" || true
        tail -100 "${stderr_path}" || true
      } >> "${monitor_path}"
      kill -TERM -- "-${pid}" 2>/dev/null || true
      sleep 10
      kill -KILL -- "-${pid}" 2>/dev/null || true
      break
    fi
    sleep 5
  done
  local returncode=0
  wait "${pid}" || returncode=$?
  local elapsed=$((SECONDS - started))
  local status=PASS
  if (( timed_out == 1 )); then
    status=TIMEOUT
    returncode=124
  elif (( returncode != 0 )); then
    status=FAIL
  fi
  printf 'END %s backend=%s sequence=%s status=%s elapsed=%ss returncode=%s\n' \
    "$(date --iso-8601=seconds)" "${backend}" "${sequence_index}" \
    "${status}" "${elapsed}" "${returncode}" >> "${monitor_path}"
  printf '{\n' > "${case_dir}/run.json"
  printf '  "schedule": "%s",\n' "${sequence_index}" >> "${case_dir}/run.json"
  printf '  "backend": "%s",\n' "${backend}" >> "${case_dir}/run.json"
  printf '  "ranks": %s,\n' "${ranks}" >> "${case_dir}/run.json"
  printf '  "grid": [%s],\n' "${grid}" >> "${case_dir}/run.json"
  printf '  "lattice": [%s],\n' "${lattice}" >> "${case_dir}/run.json"
  printf '  "status": "%s",\n' "${status}" >> "${case_dir}/run.json"
  printf '  "returncode": %s,\n' "${returncode}" >> "${case_dir}/run.json"
  printf '  "elapsed_seconds": %s,\n' "${elapsed}" >> "${case_dir}/run.json"
  printf '  "five_minute_audit": %s,\n' "$([[ ${audited} == 1 ]] && echo true || echo false)" >> "${case_dir}/run.json"
  printf '  "timeout": %s,\n' "$([[ ${timed_out} == 1 ]] && echo true || echo false)" >> "${case_dir}/run.json"
  printf '  "report": "%s",\n' "${report_path}" >> "${case_dir}/run.json"
  printf '  "solution": "%s"\n' "${solution_path}" >> "${case_dir}/run.json"
  printf '}\n' >> "${case_dir}/run.json"
  [[ ${status} == PASS ]]
}

job_started=$SECONDS
for sequence_index in "${!sequence[@]}"; do
  if (( SECONDS - job_started >= wall_limit )); then
    printf 'matrix wall limit reached before sequence %s\n' "${sequence_index}" >&2
    exit 124
  fi
  run_one "${sequence[sequence_index]}" "${sequence_index}"
done

printf 'MG_PAIR_COMPLETE root=%s elapsed=%ss\n' \
  "${result_root}" "$((SECONDS - job_started))"
