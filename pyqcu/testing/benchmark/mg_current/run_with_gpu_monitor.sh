#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
    echo "usage: bash run_with_gpu_monitor.sh OUTPUT_DIR COMMAND [ARG ...]" >&2
    exit 2
fi

bench_output_dir=$1
shift
mkdir -p "${bench_output_dir}"

bench_repo_root=$(pwd -P)
export LD_LIBRARY_PATH="${bench_repo_root}/cpp/cuda/qcu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${bench_repo_root}:${PYTHONPATH:-}"
export MPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
export QCU_LOG_DIR="${bench_output_dir}"

monitor_query="timestamp,index,uuid,name,memory.total,memory.used,utilization.gpu"
monitor_interval_ms=${GPU_MONITOR_INTERVAL_MS:-50}

nvidia-smi \
    --query-gpu="${monitor_query}" \
    --format=csv,noheader,nounits \
    > "${bench_output_dir}/baseline.csv"

nvidia-smi \
    --query-gpu="${monitor_query}" \
    --format=csv,noheader,nounits \
    -lms "${monitor_interval_ms}" \
    > "${bench_output_dir}/nvidia-smi.csv" &
monitor_pid=$!

stop_monitor() {
    if kill -0 "${monitor_pid}" 2>/dev/null; then
        kill "${monitor_pid}" 2>/dev/null || true
        wait "${monitor_pid}" 2>/dev/null || true
    fi
}
trap stop_monitor EXIT INT TERM

set +e
"$@" > "${bench_output_dir}/stdout.txt" 2> "${bench_output_dir}/stderr.txt"
command_rc=$?
set -e

stop_monitor
trap - EXIT INT TERM

awk -F',' '
function trim(value) {
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
    return value
}
FNR == NR {
    idx = trim($2)
    uuid[idx] = trim($3)
    name[idx] = trim($4)
    total[idx] = trim($5) + 0
    baseline[idx] = trim($6) + 0
    peak[idx] = baseline[idx]
    next
}
{
    idx = trim($2)
    used = trim($6) + 0
    samples[idx]++
    if (used > peak[idx]) peak[idx] = used
}
END {
    print "gpu_index\tuuid\tname\tsamples\tbaseline_mib\tpeak_mib\tdelta_mib\ttotal_mib\tpeak_fraction"
    for (idx = 0; idx < 32; idx++) {
        if (idx in baseline) {
            fraction = total[idx] > 0 ? peak[idx] / total[idx] : 0
            printf "%d\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%.6f\n", \
                idx, uuid[idx], name[idx], samples[idx] + 0, baseline[idx], \
                peak[idx], peak[idx] - baseline[idx], total[idx], fraction
        }
    }
}
' "${bench_output_dir}/baseline.csv" \
  "${bench_output_dir}/nvidia-smi.csv" \
  > "${bench_output_dir}/memory.tsv"

echo "monitor_result=${bench_output_dir}/memory.tsv command_rc=${command_rc}"
tail -n 30 "${bench_output_dir}/stdout.txt"
if [[ -s "${bench_output_dir}/stderr.txt" ]]; then
    tail -n 20 "${bench_output_dir}/stderr.txt" >&2
fi

exit "${command_rc}"
