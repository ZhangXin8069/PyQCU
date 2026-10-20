#!/usr/bin/env bash
set -euo pipefail

if [[ -n ${PYQCU_ROOT:-} ]]; then
  repo_root=$(cd -- "${PYQCU_ROOT}" && pwd)
elif [[ -n ${SLURM_SUBMIT_DIR:-} && -f ${SLURM_SUBMIT_DIR}/env.sh ]]; then
  repo_root=$(cd -- "${SLURM_SUBMIT_DIR}" && pwd)
else
  script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  repo_root=$(cd -- "${script_dir}/../../.." && pwd)
fi

source "${BSCC_ENV:-${HOME}/configure/lib/bscc-v20261015/env.sh}"
source "${repo_root}/env.sh"

export QUDA_INSTALL=${QUDA_INSTALL:-${HOME}/lib/quda-develop/build}
export QUDA_BUILD_DIR=${QUDA_BUILD_DIR:-${QUDA_INSTALL}}
export QUDA_RESOURCE_PATH=${QUDA_RESOURCE_PATH:-${repo_root}/logs/quda_resource_lime_20261021}
export QUDA_ENABLE_TUNING=0
export LD_LIBRARY_PATH=${QUDA_INSTALL}/lib:${LD_LIBRARY_PATH:-}
export OMPI_MCA_opal_warn_on_missing_libcuda=0
export OMPI_MCA_btl_openib_warn_no_device_params_found=0

lime=${PYQCU_REFERENCE_LIME:-${HOME}/data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime}
result_root=${PYQCU_LIME_RESULT_ROOT:-${repo_root}/logs/data/lime_backend_formal_20261021}
scope=${PYQCU_LIME_SCOPE:-formal}
warmups=${PYQCU_LIME_WARMUPS:-1}
repeats=${PYQCU_LIME_REPEATS:-5}
timeout_seconds=${PYQCU_LIME_TIMEOUT:-1800}

if [[ ! -f ${lime} ]]; then
  printf 'missing LIME configuration: %s\n' "${lime}" >&2
  exit 2
fi
mkdir -p "${result_root}" "${QUDA_RESOURCE_PATH}"
cd "${repo_root}"

if [[ ${PYQCU_SKIP_BUILD:-0} != 1 ]]; then
  bash ./build.sh
  bash ./install.sh
else
  test -f cpp/cuda/qcu/libqcu.so
  compgen -G 'pyqcu/cuda/qcu*.so' >/dev/null
fi

if [[ ${scope} == smoke ]]; then
  precisions=(c64)
  ranks=(1 2)
  warmups=0
  repeats=1
else
  precisions=(${PYQCU_LIME_PRECISIONS:-c64 c128})
  ranks=(${PYQCU_LIME_RANKS:-1 2 4})
fi

grid_for_ranks() {
  case "$1" in
    1) printf '%s\n' '1 1 1 1' ;;
    # Split the longest axes first to minimize each rank's halo surface.
    2) printf '%s\n' '1 1 1 2' ;;
    4) printf '%s\n' '2 1 1 2' ;;
    *) printf 'unsupported rank count: %s\n' "$1" >&2; return 2 ;;
  esac
}

run_rank_case() {
  local backend=$1
  local rank_count=$2
  local precision=$3
  local grid
  local runner
  local report
  grid=$(grid_for_ranks "${rank_count}")
  if [[ ${backend} == qcu ]]; then
    runner=pyqcu/testing/qcu/benchmark_qcu_lime_mpi.py
  elif [[ ${backend} == quda ]]; then
    runner=pyqcu/testing/quda/benchmark_quda_lime_mpi.py
  else
    printf 'unsupported backend: %s\n' "${backend}" >&2
    return 2
  fi
  mkdir -p "${result_root}/${backend}/${precision}/r${rank_count}"
  report="${result_root}/${backend}/${precision}/r${rank_count}/report.json"
  printf 'case backend=%s precision=%s ranks=%s grid=%s report=%s\n' \
    "${backend}" "${precision}" "${rank_count}" "${grid}" "${report}"
  timeout "${timeout_seconds}" mpirun \
    --allow-run-as-root --oversubscribe --bind-to none \
    -np "${rank_count}" python -B "${runner}" \
    --gauge "${lime}" --lat 24 24 24 64 --grid ${grid} \
    --dtype "${precision}" --warmups "${warmups}" --repeats "${repeats}" \
    --report "${report}"
}

for precision in "${precisions[@]}"; do
  for rank_count in "${ranks[@]}"; do
    run_rank_case qcu "${rank_count}" "${precision}"
    run_rank_case quda "${rank_count}" "${precision}"
  done
done

printf 'lime backend GPU matrix: PASS root=%s\n' "${result_root}"
