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
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export PYTHONPATH=${PYTHONPATH:-}

bscc_env=${BSCC_ENV:-${HOME}/configure/lib/bscc-v20261015/env.sh}
if [[ -f ${bscc_env} ]]; then
  # The cluster module setup used by the documented sleep.sh launcher.
  source "${bscc_env}"
fi
source "${repo_root}/env.sh"
export PYTHONPATH="${repo_root}:${PYTHONPATH}"
export QCU_STRICT_NUMERIC=1

cd "${repo_root}"
mkdir -p logs data/logs/nullvec_cache

printf 'job=%s node=%s gpus=%s\n' \
  "${SLURM_JOB_ID:-local}" "${SLURMD_NODENAME:-unknown}" \
  "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,compute_cap \
  --format=csv,noheader

python -B - <<'PY'
import torch

count = torch.cuda.device_count()
print(f"torch={torch.__version__} cuda={torch.version.cuda} devices={count}")
if count < 2:
    raise SystemExit("multi-GPU smoke requires two visible CUDA devices")
for index in range(count):
    props = torch.cuda.get_device_properties(index)
    print(f"device[{index}]={props.name} capability={props.major}.{props.minor} "
          f"memory={props.total_memory}")
PY

if [[ ${PYQCU_SKIP_BUILD:-0} != 1 ]]; then
  bash ./build.sh
  bash ./install.sh
else
  test -f cpp/cuda/qcu/libqcu.so
  compgen -G 'pyqcu/cuda/qcu*.so' >/dev/null
fi

if [[ ${PYQCU_BENCH_ONLY:-0} != 1 ]]; then
  python -B pyqcu/testing/qcu/single_qcu_api.py
  python -B pyqcu/testing/qcu/single_qcu_wilson_dslash.py \
    --lat 8 8 8 16 --no-progress
  python -B pyqcu/testing/qcu/single_qcu_clover_bistabcg.py \
    --lat 8 8 8 16 --no-progress

  python -B - <<'PY'
import json
import torch

from pyqcu.testing import test_multi_gpu_multigrid

_results, consistency = test_multi_gpu_multigrid(
    nthreads=2, lat_size=[8, 8, 8, 16], mass=0.05, atol=1e-6,
    num_levels=2, tol=3e-5, verbose=True)
print("multi_gpu_consistency=" + json.dumps(consistency, sort_keys=True))
PY
fi

python -B pyqcu/testing/qcu/bench_multi_gpu.py \
  --lat 8x8x8x16 --nthreads 2 --device-ids 0 1 \
  --warmups 1 --repeats 5 \
  --cache-dir data/logs/nullvec_cache \
  --output "logs/cuda_multi_gpu_${SLURM_JOB_ID:-local}.json"
python -B pyqcu/testing/qcu/bench_multi_gpu.py \
  --lat 8x8x8x16 --nthreads 2 --device-ids 0 1 --no-reference \
  --warmups 1 --repeats 5 --cache-dir data/logs/nullvec_cache \
  --output "logs/cuda_multi_gpu_noref_${SLURM_JOB_ID:-local}.json"

lime=${PYQCU_REFERENCE_LIME:-${HOME}/data/beta6.20_mu-0.2770_ms-0.2310_L24x64_cfg_9950.lime}
if [[ -f ${lime} && ${PYQCU_SKIP_LIME:-0} != 1 ]]; then
  python -B pyqcu/testing/simple/qcu_bistabcg.py \
    --lat 24 24 24 64 --gauge "${lime}" --atol 1e-6 \
    --max-iter 500 --device cuda:0 \
    --report "logs/cuda_lime24x64_c64_${SLURM_JOB_ID:-local}.json"
else
  printf 'LIME regression skipped: %s\n' "${lime}"
fi

printf 'multi-GPU smoke: PASS\n'
