#!/usr/bin/env bash
# Flat CUDA Legacy example: two MPI processes, two visible devices, two levels.
#
# Required input: set PYQCU_GAUGE (LIME/HDF5) or PYQCU_LIME.
# Optional: PYQCU_SOURCE, PYQCU_COARSE_CACHE, PYQCU_WORK_DIR, PYQCU_OUTPUT,
#           PYQCU_REPORT, PYQCU_RANKS, PYQCU_GRID, PYQCU_DTYPE, PYQCU_ATOL,
#           PYQCU_OMP_THREADS, PYQCU_LIBQCU, PYQCU_MPIRUN, PYQCU_PYTHON.
# DRY_RUN=1 prints only the resolved command.
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(CDPATH= cd -- "${script_dir}/../../../.." && pwd)
cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

python_bin=${PYQCU_PYTHON:-python3}
rank_checker="${script_dir}/mpi_ranks_power_of_two.py"
solve_case="${script_dir}/solve_case.py"
[[ -f ${rank_checker} ]] || die "rank validator not found: ${rank_checker}"
[[ -f ${solve_case} ]] || die "solve_case.py not found: ${solve_case}"

ranks=${PYQCU_RANKS:-2}
"${python_bin}" "${rank_checker}" --quiet "${ranks}"

gauge=${PYQCU_GAUGE:-}
lime=${PYQCU_LIME:-}
source_path=${PYQCU_SOURCE:-}
coarse_cache=${PYQCU_COARSE_CACHE:-}
work_dir=${PYQCU_WORK_DIR:-}
output=${PYQCU_OUTPUT:-}
report=${PYQCU_REPORT:-}
dtype=${PYQCU_DTYPE:-c64}
atol=${PYQCU_ATOL:-1e-6}
omp_threads=${PYQCU_OMP_THREADS:-1}
libqcu=${PYQCU_LIBQCU:-auto}
mpirun_bin=${PYQCU_MPIRUN:-mpirun}
dry_run=${DRY_RUN:-0}

case "${ranks}" in
  2) default_grid='1 2 1 1' ;;
  4) default_grid='1 4 1 1' ;;
  *) default_grid='' ;;
esac
grid_spec=${PYQCU_GRID:-${default_grid}}
grid_args=()
if [[ -n ${grid_spec} ]]; then
  read -r -a grid_args <<< "${grid_spec}"
  ((${#grid_args[@]} == 4)) || die "PYQCU_GRID must contain four integers"
fi

if [[ ${dry_run} == 1 ]]; then
  if [[ -z ${gauge} && -z ${lime} ]]; then
    gauge='<set PYQCU_GAUGE or PYQCU_LIME>'
  fi
else
  if [[ -n ${gauge} ]]; then
    [[ -f ${gauge} ]] || die "PYQCU_GAUGE file not found: ${gauge}"
  elif [[ -n ${lime} ]]; then
    [[ -f ${lime} ]] || die "PYQCU_LIME file not found: ${lime}"
  else
    die 'missing gauge input; set PYQCU_GAUGE (LIME/HDF5) or PYQCU_LIME'
  fi
  if [[ -n ${source_path} ]]; then
    [[ -f ${source_path} ]] || die "PYQCU_SOURCE file not found: ${source_path}"
  fi
  if [[ -n ${coarse_cache} ]]; then
    [[ -f ${coarse_cache} ]] || \
      die "PYQCU_COARSE_CACHE file not found: ${coarse_cache}"
  fi
  if [[ -n ${CUDA_VISIBLE_DEVICES:-} ]]; then
    visible_count=0
    IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
    for device in "${visible_devices[@]}"; do
      [[ -z ${device} ]] || ((visible_count += 1))
    done
    ((visible_count >= ranks)) || die \
      "CUDA_VISIBLE_DEVICES exposes ${visible_count} device(s), fewer than ${ranks} ranks"
  fi
fi

command=(
  "${python_bin}" "${solve_case}"
  --dtype "${dtype}"
  --atol "${atol}"
  --omp-threads "${omp_threads}"
  --libqcu "${libqcu}"
  --mpirun "${mpirun_bin}"
  --lat 8 8 8 16
  --dof 12 12
  --mg-grid 2 2 2 2
)
if ((${#grid_args[@]})); then
  command+=(--grid "${grid_args[@]}")
fi
if [[ -n ${gauge} ]]; then
  command+=(--gauge "${gauge}")
elif [[ -n ${lime} ]]; then
  command+=(--lime "${lime}")
fi
[[ -z ${source_path} ]] || command+=(--source "${source_path}")
[[ -z ${coarse_cache} ]] || command+=(--coarse-cache "${coarse_cache}")
[[ -z ${work_dir} ]] || command+=(--work-dir "${work_dir}")
[[ -z ${output} ]] || command+=(--output "${output}")
[[ -z ${report} ]] || command+=(--report "${report}")

# Keep the documented backend/depth/rank semantics fixed even when callers
# override other solve_case options through positional arguments.
command+=("$@")
command+=(--backend cuda --ranks "${ranks}" --levels 2)

if [[ ${dry_run} == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

exec "${command[@]}"
