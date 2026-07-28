# BUGFIX 2026-07-28 R3: add set -e so build failures stop the pipeline.
set -e
echo "=== PyQCU build.sh ==="
source ./env.sh
pushd ./cpp/cuda/qcu
bash ./make.sh
popd
echo "=== PyQCU build.sh: SUCCESS ==="