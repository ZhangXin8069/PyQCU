# BUGFIX 2026-07-28 R3: add set -e and && chaining for error detection.
set -euo pipefail

source ./env.sh
repo_root=$(realpath ../../..)
source_dir=$(pwd)
build_dir="${repo_root}/data/build/cuda-qcu-dcu"

ln -sf CMakeLists-dcu.txt CMakeLists.txt
cuda_args=()
if [[ -n "${ROCM_PATH:-}" && -x "${ROCM_PATH}/cuda/bin/nvcc" ]]; then
  cuda_args+=("-DCMAKE_CUDA_COMPILER=${ROCM_PATH}/cuda/bin/nvcc")
fi

cmake -S "${source_dir}" -B "${build_dir}" \
  -DCMAKE_LIBRARY_OUTPUT_DIRECTORY="${source_dir}" "${cuda_args[@]}"
cmake --build "${build_dir}" --parallel "$(nproc)"
echo "make.sh: SUCCESS"
