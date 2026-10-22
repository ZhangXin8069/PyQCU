#!/usr/bin/env bash
# Experimental CUDA Legacy example: two MPI processes and three levels.
#
# Required inputs: set PYQCU_GAUGE (LIME/HDF5) or PYQCU_LIME, plus
# PYQCU_COARSE_CACHE and PYQCU_COARSE_CACHE_LEVEL for both coarse transitions.
# Optional: PYQCU_SOURCE, PYQCU_WORK_DIR, PYQCU_OUTPUT, PYQCU_REPORT,
#           PYQCU_RANKS, PYQCU_GRID, PYQCU_DTYPE, PYQCU_ATOL,
#           PYQCU_OMP_THREADS, PYQCU_LIBQCU, PYQCU_MPIRUN, PYQCU_PYTHON.
# DRY_RUN=1 prints only the resolved command; warnings go to stderr.
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(CDPATH= cd -- "${script_dir}/../../../.." && pwd)
cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

printf '%s\n' \
  'Legacy default depth: 2 levels.' \
  'WARNING: CUDA Legacy 3-level execution is experimental; use it only for controlled validation.' \
  'The independent distributed true-residual check remains enabled.' >&2

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
coarse_cache_level=${PYQCU_COARSE_CACHE_LEVEL:-}
work_dir=${PYQCU_WORK_DIR:-}
output=${PYQCU_OUTPUT:-}
report=${PYQCU_REPORT:-}
dtype=${PYQCU_DTYPE:-c64}
atol=${PYQCU_ATOL:-1e-6}
omp_threads=${PYQCU_OMP_THREADS:-1}
libqcu=${PYQCU_LIBQCU:-auto}
mpirun_bin=${PYQCU_MPIRUN:-mpirun}
dry_run=${DRY_RUN:-0}

for arg in "$@"; do
  case "${arg}" in
    --backend|--backend=*|--ranks|--ranks=*|--levels|--levels=*)
      die 'backend, rank count and level count are controlled by this example'
      ;;
    --coarse-cache|--coarse-cache=*|--coarse-cache-level|--coarse-cache-level=*)
      die 'coarse caches are controlled by PYQCU_COARSE_CACHE and PYQCU_COARSE_CACHE_LEVEL'
      ;;
    --skip-independent-residual|--skip-independent-residual=*)
      die 'the 3-level example requires the independent true-residual check'
      ;;
  esac
done

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
  [[ -n ${coarse_cache} ]] || coarse_cache='<set PYQCU_COARSE_CACHE>'
  [[ -n ${coarse_cache_level} ]] || \
    coarse_cache_level='<set PYQCU_COARSE_CACHE_LEVEL>'
else
  if [[ -n ${gauge} ]]; then
    [[ -f ${gauge} ]] || die "PYQCU_GAUGE file not found: ${gauge}"
  elif [[ -n ${lime} ]]; then
    [[ -f ${lime} ]] || die "PYQCU_LIME file not found: ${lime}"
  else
    die 'missing gauge input; set PYQCU_GAUGE (LIME/HDF5) or PYQCU_LIME'
  fi
  [[ -n ${coarse_cache} ]] || \
    die 'missing PYQCU_COARSE_CACHE for the first coarse transition'
  [[ -n ${coarse_cache_level} ]] || \
    die 'missing PYQCU_COARSE_CACHE_LEVEL for the second coarse transition'
  [[ -f ${coarse_cache} ]] || \
    die "PYQCU_COARSE_CACHE file not found: ${coarse_cache}"
  [[ -f ${coarse_cache_level} ]] || \
    die "PYQCU_COARSE_CACHE_LEVEL file not found: ${coarse_cache_level}"
  if [[ -n ${source_path} ]]; then
    [[ -f ${source_path} ]] || die "PYQCU_SOURCE file not found: ${source_path}"
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
  --dof 12 12 12
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
[[ -z ${work_dir} ]] || command+=(--work-dir "${work_dir}")
[[ -z ${output} ]] || command+=(--output "${output}")
[[ -z ${report} ]] || command+=(--report "${report}")
command+=("$@")
command+=(
  --backend cuda
  --ranks "${ranks}"
  --levels 3
  --coarse-cache "${coarse_cache}"
  --coarse-cache-level "${coarse_cache_level}"
)

if [[ ${dry_run} == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

exec "${command[@]}"
