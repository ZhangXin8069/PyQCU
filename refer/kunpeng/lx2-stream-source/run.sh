#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=FALSE
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0
exec python3 -B run_stream.py "$@"
