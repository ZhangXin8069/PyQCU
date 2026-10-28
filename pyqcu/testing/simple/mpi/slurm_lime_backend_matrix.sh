#!/usr/bin/env bash
set -euo pipefail

# Run inside one Slurm allocation. The Python matrix has a stricter 570 s
# internal deadline so a 10-minute Slurm allocation still has time to publish
# the manifest and fail-closed summary.
dry_run=${DRY_RUN:-0}
if [[ -n ${PYQCU_ROOT:-} ]]; then
  repo_root=$(cd -- "${PYQCU_ROOT}" && pwd)
elif [[ -n ${SLURM_SUBMIT_DIR:-} && -f ${SLURM_SUBMIT_DIR}/env.sh ]]; then
  repo_root=$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)
else
  script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  repo_root=$(cd -- "${script_dir}/../../../.." && pwd)
fi

site_env=${PYQCU_SLURM_ENV:-${BSCC_ENV:-}}
if [[ -n ${site_env} ]]; then
  # shellcheck disable=SC1090
  source "${site_env}"
fi
if [[ -f ${repo_root}/env.sh ]]; then
  # shellcheck disable=SC1091
  source "${repo_root}/env.sh"
fi

for env_name in $(compgen -e); do
  if [[ ${env_name} == *TRACE* ]]; then
    unset "${env_name}"
  fi
done
export QUDA_ENABLE_TUNING=0
export PYQCU_MPI_DEVICE_AWARE=${PYQCU_MPI_DEVICE_AWARE:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
unset OMP_PROC_BIND OMP_PLACES

lime=${PYQCU_REFERENCE_LIME:-${HOME}/data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime}
result_root=${PYQCU_LIME_RESULT_ROOT:-${repo_root}/data/cuda_quda_matrix/v$(date +%Y%m%d%H%M%S)}
lattice=${PYQCU_LIME_LATTICE:-"24 24 24 64"}
cases=${PYQCU_LIME_CASES:-"1:1,1,1,1 2:2,1,1,1 4:2,2,1,1"}
backends=${PYQCU_LIME_BACKENDS:-"qcu quda"}
precision=${PYQCU_LIME_PRECISION:-${PYQCU_LIME_PRECISIONS:-c64}}
max_iter=${PYQCU_LIME_MAX_ITER:-1000}
atol=${PYQCU_LIME_ATOL:-1e-6}
seed=${PYQCU_LIME_SEED:-42}
wall_limit=${PYQCU_LIME_WALL_LIMIT:-570}
slurm_time_limit=${PYQCU_LIME_SLURM_TIME_LIMIT:-600}
child_timeout=${PYQCU_LIME_CHILD_TIMEOUT:-180}
mpirun=${PYQCU_MPIRUN:-mpirun}
python_bin=${PYQCU_PYTHON:-python3}
defer_residual=${PYQCU_LIME_DEFER_RESIDUAL:-1}
rank_helper=${repo_root}/pyqcu/testing/simple/mpi/mpi_ranks_power_of_two.py

scope=${PYQCU_LIME_SCOPE:-formal}
if [[ ${scope} == smoke ]]; then
  warmups=${PYQCU_LIME_WARMUPS:-0}
  repeats=${PYQCU_LIME_REPEATS:-1}
else
  warmups=${PYQCU_LIME_WARMUPS:-1}
  repeats=${PYQCU_LIME_REPEATS:-3}
fi

if [[ ${precision} != c64 && ${precision} != c128 ]]; then
  printf 'one precision per bounded job is required: %s\n' "${precision}" >&2
  exit 2
fi
if [[ ! ${wall_limit} =~ ^[0-9]+$ || ! ${slurm_time_limit} =~ ^[0-9]+$ ]]; then
  printf 'wall limits must be integer seconds\n' >&2
  exit 2
fi
if [[ ${defer_residual} != 0 && ${defer_residual} != 1 ]]; then
  printf 'PYQCU_LIME_DEFER_RESIDUAL must be 0 or 1\n' >&2
  exit 2
fi
if (( wall_limit <= 0 || slurm_time_limit <= wall_limit ||
      slurm_time_limit > 600 )); then
  printf 'require 0 < wall-limit < slurm-time-limit <= 600\n' >&2
  exit 2
fi

if [[ ! -f ${lime} && ${dry_run} != 1 ]]; then
  printf 'missing LIME configuration: %s\n' "${lime}" >&2
  exit 2
fi

if [[ ${PYQCU_SKIP_BUILD:-0} != 1 && ${dry_run} != 1 ]]; then
  (cd "${repo_root}" && bash ./build.sh && bash ./install.sh)
elif [[ ${dry_run} != 1 ]]; then
  test -f "${repo_root}/cpp/cuda/qcu/libqcu.so"
  compgen -G "${repo_root}/pyqcu/cuda/qcu*.so" >/dev/null
fi

parse_case() {
  local entry=$1
  local rank_part=${entry%%:*}
  local grid_part=${entry#*:}
  if [[ ${rank_part} == "${entry}" || -z ${grid_part} ]]; then
    printf 'invalid case %s; expected rank:gx,gy,gz,gt\n' "${entry}" >&2
    return 2
  fi
  IFS=',' read -r gx gy gz gt <<< "${grid_part}"
  if [[ ! ${rank_part} =~ ^[0-9]+$ || ! ${gx} =~ ^[0-9]+$ ||
        ! ${gy} =~ ^[0-9]+$ || ! ${gz} =~ ^[0-9]+$ ||
        ! ${gt} =~ ^[0-9]+$ ]]; then
    printf 'invalid case %s; rank and grid must be positive integers\n' \
      "${entry}" >&2
    return 2
  fi
  if (( gx * gy * gz * gt != rank_part )); then
    printf 'case %s grid product %d != ranks %d\n' \
      "${entry}" "$((gx * gy * gz * gt))" "${rank_part}" >&2
    return 2
  fi
  "${python_bin}" "${rank_helper}" --quiet "${rank_part}"
  printf '%s %s %s %s %s\n' "${rank_part}" "${gx}" "${gy}" "${gz}" "${gt}"
}

selected_backends=()
for backend in ${backends}; do
  case ${backend} in
    qcu) selected_backends+=(cuda) ;;
    quda) selected_backends+=(quda) ;;
    *) printf 'unsupported backend: %s\n' "${backend}" >&2; exit 2 ;;
  esac
done
if (( ${#selected_backends[@]} == 0 )); then
  printf 'no backends selected\n' >&2
  exit 2
fi
backend_csv=$(IFS=,; printf '%s' "${selected_backends[*]}")

matrix_args=(
  "${python_bin}" -B "${repo_root}/pyqcu/testing/simple/mpi/compare_three_way.py"
  --matrix --backends "${backend_csv}" --gauge "${lime}"
  --lat ${lattice} --dtype "${precision}" --atol "${atol}"
  --max-iter "${max_iter}" --warmups "${warmups}" --repeats "${repeats}"
  --seed "${seed}" --result-root "${result_root}"
  --wall-limit "${wall_limit}" --slurm-time-limit "${slurm_time_limit}"
  --child-timeout "${child_timeout}" --mpirun "${mpirun}"
)
if [[ ${defer_residual} == 1 ]]; then
  matrix_args+=(--defer-independent-residual)
fi
if [[ -n ${PYQCU_LIME_SOURCE:-} ]]; then
  matrix_args+=(--source "${PYQCU_LIME_SOURCE}")
fi
if [[ -n ${PYQCU_LIME_MASS:-} ]]; then
  matrix_args+=(--mass "${PYQCU_LIME_MASS}")
fi
if [[ -n ${PYQCU_LIME_CSW:-} ]]; then
  matrix_args+=(--csw "${PYQCU_LIME_CSW}")
fi
for case_entry in ${cases}; do
  parse_case "${case_entry}" >/dev/null
  matrix_args+=(--rank-case "${case_entry}")
  IFS=',' read -r gx gy gz gt <<< "${case_entry#*:}"
  printf 'matrix case grid: --grid %s %s %s %s\n' "${gx}" "${gy}" "${gz}" "${gt}"
done
if [[ ${dry_run} == 1 ]]; then
  matrix_args+=(--dry-run)
fi

printf 'host=%s job=%s scope=%s ranks_cases=%s backends=%s dtype=%s wall=%ss/%ss\n' \
  "$(hostname)" "${SLURM_JOB_ID:-}" "${scope}" "${cases}" \
  "${backend_csv}" "${precision}" "${wall_limit}" "${slurm_time_limit}"

status=0
if [[ ${dry_run} == 1 ]]; then
  (cd "${repo_root}" && "${matrix_args[@]}") || status=$?
else
  (cd "${repo_root}" && timeout --signal=TERM --kill-after=15 \
    "${slurm_time_limit}" "${matrix_args[@]}") || status=$?
fi

if [[ ${status} == 0 ]]; then
  if [[ ${dry_run} == 1 ]]; then
    printf 'lime backend MPI matrix: DRY-RUN root=%s trace=%s tuning=%s\n' \
      "${result_root}" disabled "${QUDA_ENABLE_TUNING}"
  else
    printf 'lime backend MPI matrix: GPU-COMPLETE root=%s manifest=%s summary=%s independent_residual=CPU-PENDING\n' \
      "${result_root}" "${result_root}/run-manifest.json" \
      "${result_root}/summary.json"
  fi
elif [[ ${status} == 124 || ${status} == 137 ]]; then
  printf 'lime backend MPI matrix: TIMEOUT after %ss; no speedup published\n' \
    "${slurm_time_limit}" >&2
else
  printf 'lime backend MPI matrix: FAIL status=%s root=%s\n' \
    "${status}" "${result_root}" >&2
fi
exit "${status}"
