#!/usr/bin/env bash
set -euo pipefail

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
  if [[ ${env_name} =~ ^(PYQCU|QUDA)_.*TRACE ]]; then
    unset "${env_name}"
  fi
done
export QUDA_ENABLE_TUNING=0
export PYQCU_MPI_DEVICE_AWARE=${PYQCU_MPI_DEVICE_AWARE:-0}
export KUPL_NUMA_BIND=0
unset OMP_PROC_BIND OMP_PLACES

lime=${PYQCU_REFERENCE_LIME:-${HOME}/data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime}
result_root=${PYQCU_LIME_RESULT_ROOT:-${repo_root}/data/lime_backend_matrix}
lattice=${PYQCU_LIME_LATTICE:-"24 24 24 64"}
warmups=${PYQCU_LIME_WARMUPS:-1}
repeats=${PYQCU_LIME_REPEATS:-5}
max_iter=${PYQCU_LIME_MAX_ITER:-1000}
timeout_seconds=${PYQCU_LIME_TIMEOUT:-1800}
power_limit_watts=${PYQCU_LIME_POWER_LIMIT_WATTS:-}
power_check=${PYQCU_LIME_POWER_CHECK:-1}
mpirun=${PYQCU_MPIRUN:-mpirun}
python_bin=${PYQCU_PYTHON:-python3}
rank_helper=${repo_root}/pyqcu/testing/simple/mpi/mpi_ranks_power_of_two.py

scope=${PYQCU_LIME_SCOPE:-formal}
if [[ ${scope} == smoke ]]; then
  backends=(${PYQCU_LIME_BACKENDS:-qcu quda})
  precisions=(${PYQCU_LIME_PRECISIONS:-c64})
  cases=(${PYQCU_LIME_CASES:-1:1,1,1,1 2:1,1,1,2})
  warmups=0
  repeats=1
else
  backends=(${PYQCU_LIME_BACKENDS:-qcu quda})
  precisions=(${PYQCU_LIME_PRECISIONS:-c64 c128})
  cases=(${PYQCU_LIME_CASES:-1:1,1,1,1 2:1,1,1,2 4:2,1,1,2})
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

check_power() {
  [[ -n ${power_limit_watts} && ${power_check} == 1 ]] || return 0
  if [[ ${dry_run} == 1 ]]; then
    return 0
  fi
  command -v nvidia-smi >/dev/null 2>&1 || {
    printf 'power limit set but nvidia-smi is unavailable\n' >&2
    return 2
  }
  local draw
  draw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits |
    awk 'BEGIN {max=0} {if ($1+0 > max) max=$1+0} END {print max}')
  awk -v draw="${draw}" -v limit="${power_limit_watts}" \
    'BEGIN {exit !(draw <= limit)}' || {
      printf 'GPU power draw %.2f W exceeds configured limit %.2f W\n' \
        "${draw}" "${power_limit_watts}" >&2
      return 2
    }
}

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

run_case() {
  local backend=$1
  local precision=$2
  local rank_count=$3
  local gx=$4
  local gy=$5
  local gz=$6
  local gt=$7
  local runner
  local report_dir
  local report
  local staged_report
  local returncode
  local -a command
  case ${backend} in
    qcu) runner=pyqcu/testing/qcu/benchmark_qcu_lime_mpi.py ;;
    quda) runner=pyqcu/testing/quda/benchmark_quda_lime_mpi.py ;;
    *)
      printf 'unsupported backend: %s\n' "${backend}" >&2
      return 2
      ;;
  esac
  report_dir=${result_root}/${backend}/${precision}/r${rank_count}
  report=${report_dir}/report.json
  staged_report=${report_dir}/.report.${BASHPID}.partial.json
  command=(
    "${mpirun}" --allow-run-as-root --oversubscribe --bind-to none
    -np "${rank_count}" "${python_bin}" -B "${repo_root}/${runner}"
    --gauge "${lime}" --lat ${lattice} --grid "${gx}" "${gy}" "${gz}" "${gt}"
    --dtype "${precision}" --warmups "${warmups}" --repeats "${repeats}"
    --max-iter "${max_iter}" --report "${staged_report}"
  )
  if [[ ${dry_run} == 1 ]]; then
    printf '%q ' "${command[@]:0:${#command[@]}-2}" --report "${report}"
    printf '\n'
    return 0
  fi
  check_power
  mkdir -p "${report_dir}"
  printf 'case backend=%s precision=%s ranks=%s grid=%s,%s,%s,%s report=%s\n' \
    "${backend}" "${precision}" "${rank_count}" \
    "${gx}" "${gy}" "${gz}" "${gt}" "${report}"
  if [[ ${timeout_seconds} == 0 ]]; then
    if (cd "${repo_root}" && "${command[@]}"); then
      returncode=0
    else
      returncode=$?
    fi
  else
    if (cd "${repo_root}" && timeout "${timeout_seconds}" "${command[@]}"); then
      returncode=0
    else
      returncode=$?
    fi
  fi
  if [[ -f ${staged_report} ]]; then
    mv -f "${staged_report}" "${report}"
  fi
  return "${returncode}"
}

for case_entry in "${cases[@]}"; do
  read -r rank_count gx gy gz gt < <(parse_case "${case_entry}")
  for backend in "${backends[@]}"; do
    for precision in "${precisions[@]}"; do
      run_case "${backend}" "${precision}" \
        "${rank_count}" "${gx}" "${gy}" "${gz}" "${gt}"
    done
  done
done

if [[ ${dry_run} == 1 ]]; then
  printf 'lime backend MPI matrix: DRY-RUN root=%s trace=%s tuning=%s\n' \
    "${result_root}" disabled "${QUDA_ENABLE_TUNING}"
else
  printf 'lime backend MPI matrix: PASS root=%s trace=%s tuning=%s\n' \
    "${result_root}" disabled "${QUDA_ENABLE_TUNING}"
fi
