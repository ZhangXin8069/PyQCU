#!/bin/bash
# Submit one KUPL-QCU Kunpeng matrix group to the LX2 scheduler.
#
# The Python driver launches mpirun itself, so each dsub job is one task that
# owns a whole node allocation.  Use one job per node/shard:
#
#   ./submit_kunpeng_all.sh --mode prepare --tag kall_prep --ranks 32 --threads 18
#   ./submit_kunpeng_all.sh --mode solve --tag kall_l24_a --ranks 32 --threads 18 \
#       --shard-index 0 --shard-count 2 --node cn19699 --extra "--expand ..."
#
# `--extra` is appended verbatim to the Python driver command line.
#
# Observations from the 2026-11-17 remote runs (context only, no behaviour):
#   * dsub semantics: -nn is the *node* count, -N the *total* replica count and
#     -rpn the replicas per node, so the single-node request below is
#     `-nn 1 -N $RANKS -rpn $RANKS`.  Two nodes would be `-nn 2 -N 32 -rpn 16`,
#     and the rankfile derived from $CCS_ALLOC_FILE must actually be passed to
#     mpirun (--rankfile): writing the file alone changes no placement.
#   * Two-node route verified on 2026-11-17 22:15 (job 2155536): a cluster-local
#     SSH key is accepted by the compute-node sshd (`Server accepts key`) but the
#     session is then closed, so `plm_rsh` is NOT usable here.  The working path
#     is `CCS_DRUN_RANK_FILE` + `/opt/batch/agent/tools/drun -mpi pmix`; `drun`
#     warns that -nn/-rpn/-R are ignored when the rankfile env is set.  Details:
#     docs/pyqcu_test_l48_remote.md sec. 0.1.6.
#   * L48 32 ranks on one exclusive node (-x job) solved the BiCGStab control in
#     1.665 s / 11 iterations, while 16 ranks on a shared node needed 183 s for
#     the same work.  Keep the exclusive-node default and never compare a shared
#     point with an exclusive one.
#   * UCG "version is incompatible" lines are non-fatal (one pair per rank:
#     HMPI expects UCG 26.1, the host provides 26.0.RC1 and falls back); the
#     job rc, not those lines, decides failure.
#   * Environment: HPCKit HMPI plus bisheng libomp and a gcc libstdc++ that
#     provides GLIBCXX_3.4.29, on Python 3.10.9 (anaconda 2023.3, h5py without
#     MPI).  dsub/djob only work from a login shell (see the submit tail).
#
# Scope of the L48 work: the CG group is blocked (no CG entry in any L48-capable
# harness, `applyWilsonCgQcu` being Wilson-only) and the two-node 32r row never
# allocated, so only BiCGStab/MG rows are meaningful here; Strict-MG is out of
# scope for these scripts.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PY_SCRIPT="$SCRIPT_DIR/test_kunpeng_all.py"
DSUB="${DSUB:-/opt/batch/cli/bin/dsub}"
OUTDIR="${OUTDIR:-$REPO/data/kunpeng_all_20261115}"

MODE=""
TAG=""
RANKS=32
THREADS=18
NODE=""
SHARD_INDEX=0
SHARD_COUNT=1
LIMIT=""
EXTRA=""
WALLTIME=900
DRY_RUN=0
NO_SUBMIT=0
TASK_MODEL=mpi
EXCLUSIVE=1

usage() {
  cat <<'EOF'
Usage: submit_kunpeng_all.sh --mode prepare|prewarm|solve|merge|validate [options]

  --mode MODE          prepare | prewarm | solve | merge | validate
                       (prepare/solve go to the scheduler; prewarm/merge/validate
                        run in the current shell, no allocation needed)
  --tag TAG            job/tag name (default: kall_<mode>_<pid>)
  --ranks N            MPI ranks per case (default 32)
  --threads N          OpenMP threads per rank (default 18)
  --node NAME          optional scheduler node preference (-nl)
  --shard-index N      shard index (default 0)
  --shard-count N      shard count (default 1)
  --limit N            max cases in this job
  --walltime SEC       scheduler wall time (default 900)
  --outdir DIR         output root (default: data/kunpeng_all_20261115)
  --task-model MODEL   mpi (default, -N ranks) | single (-N 1, whole-node CPU)
  --shared             omit -x job (allow a partially used node)
  --extra "ARGS"       appended to the Python driver command line
  --dry-run            print the dsub command, do not submit
  -h, --help           this help
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --ranks) RANKS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --node) NODE="$2"; shift 2 ;;
    --shard-index) SHARD_INDEX="$2"; shift 2 ;;
    --shard-count) SHARD_COUNT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --walltime) WALLTIME="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --task-model) TASK_MODEL="$2"; shift 2 ;;
    --shared) EXCLUSIVE=0; shift ;;
    --extra) EXTRA="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "$MODE" ]; then
  echo "--mode is required" >&2
  usage >&2
  exit 2
fi
case "$MODE" in prepare|prewarm|solve|merge|validate) ;; *) echo "invalid --mode: $MODE" >&2; exit 2 ;; esac
if [ -z "$TAG" ]; then
  TAG="kall_${MODE}_$$"
fi
if ! [[ "$RANKS" =~ ^[0-9]+$ ]] || ! [[ "$THREADS" =~ ^[0-9]+$ ]]; then
  echo "--ranks/--threads must be integers" >&2
  exit 2
fi
if [ $((RANKS * THREADS)) -gt 608 ]; then
  echo "ranks*threads=$((RANKS * THREADS)) exceeds 608; skip (use a smaller pair)" >&2
  exit 2
fi
TOTAL_CPU=$((RANKS * THREADS))
case "$TASK_MODEL" in mpi|single) ;; *) echo "invalid --task-model: $TASK_MODEL" >&2; exit 2 ;; esac

mkdir -p "$OUTDIR"/{logs,jobs,rows,caches,reports,solutions,sources,gauges}

PY_ARGS=(--mode "$MODE" --outdir "$OUTDIR" --ranks "$RANKS" --threads "$THREADS"
         --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT")
if [ -n "$LIMIT" ]; then PY_ARGS+=(--limit "$LIMIT"); fi
if [ -n "$EXTRA" ]; then
  # shellcheck disable=SC2206
  read -r -a EXTRA_ARR <<< "$EXTRA"
  PY_ARGS+=("${EXTRA_ARR[@]}")
fi

if [ "$MODE" = "merge" ] || [ "$MODE" = "prewarm" ] || [ "$MODE" = "validate" ]; then
  # merge/prewarm/validate never need a whole-node allocation: run them here.
  # Same environment as the dsub job scripts so h5py/numpy come from the
  # HPCKit anaconda (the system python3 has none of them).
  export PATH=/work_ssd/software/HPCKit/25.2.1/hmpi/bisheng/release/hmpi/bin:/work_ssd/software/soft/tool/anaconda3/2023.3/bin:$PATH
  export LD_LIBRARY_PATH=/work_ssd/software/HPCKit/25.2.1/hmpi/bisheng/release/hmpi/lib:/work_ssd/software/HPCKit/alllib:${LD_LIBRARY_PATH-}
  export PYTHONPATH=$REPO:${PYTHONPATH-}
  exec python3 -B "$PY_SCRIPT" "${PY_ARGS[@]}"
fi

QUOTED="$(printf '%q ' "${PY_ARGS[@]}")"
JOB="$OUTDIR/jobs/$TAG.sh"
{
  echo '#!/bin/bash'
  if [ "$TASK_MODEL" = "single" ]; then
    echo '#DSUB -q q_hpcapp'
    echo "#DSUB -n $TAG"
    echo '#DSUB -nn 1'
    echo '#DSUB -N 1'
    echo '#DSUB -rpn 1'
    echo "#DSUB -R \"cpu=$TOTAL_CPU;mem=400G\""
  else
    echo '#DSUB --mpi hmpi'
    echo '#DSUB -q q_hpcapp'
    echo "#DSUB -n $TAG"
    echo '#DSUB -nn 1'
    echo "#DSUB -N $RANKS"
    echo "#DSUB -rpn $RANKS"
    echo "#DSUB -R \"cpu=$THREADS;mem=8G\""
  fi
  # Exclusive by default: a 32r L48 job needs the whole node (see header).
  if [ "$EXCLUSIVE" = "1" ]; then echo '#DSUB -x job'; fi
  echo "#DSUB -oo $OUTDIR/logs/$TAG.out"
  echo "#DSUB -eo $OUTDIR/logs/$TAG.err"
  echo '#DSUB -T '"$WALLTIME"
  if [ -n "$NODE" ]; then echo "#DSUB -nl $NODE"; fi
  echo 'set +u'
  echo 'source /work_ssd/software/HPCKit/25.2.1/setvars.sh >/dev/null 2>&1 || true'
  echo 'set -u'
  echo 'export PATH=/work_ssd/software/HPCKit/25.2.1/hmpi/bisheng/release/hmpi/bin:/work_ssd/software/soft/tool/anaconda3/2023.3/bin:$PATH'
  echo 'export LD_LIBRARY_PATH=/work_ssd/software/HPCKit/25.2.1/hmpi/bisheng/release/hmpi/lib:/work_ssd/software/HPCKit/alllib:${LD_LIBRARY_PATH-}'
  echo "export PYTHONPATH=$REPO:\${PYTHONPATH-}"
  echo 'export OMPI_MCA_rmaps_base_oversubscribe=1'
  echo "cd $REPO"
  echo "exec python3 -B $PY_SCRIPT $QUOTED"
} > "$JOB"
chmod +x "$JOB"

echo "job_script=$JOB"
echo "tag=$TAG mode=$MODE ranks=$RANKS threads=$THREADS shard=$SHARD_INDEX/$SHARD_COUNT node=${NODE:-auto}"

if [ "$DRY_RUN" = "1" ]; then
  echo "dsub -q q_hpcapp $JOB"
  exit 0
fi

if [ ! -x "$DSUB" ]; then
  echo "dsub not found at $DSUB; run on login06 or set DSUB=" >&2
  exit 2
fi

# Current dsub parses #DSUB resource blocks only with -s/--script; use a login
# shell so the batch CLI environment and license variables are present.  Poll
# with `bash -lc 'djob -ll <id>'` for the same reason.
DSUB_OUT="$(bash -lc "dsub -s '$JOB'" 2>&1)"
JOB_ID="$(printf '%s\n' "$DSUB_OUT" | grep -o '[0-9][0-9]*' | head -1)"
if [ -z "$JOB_ID" ]; then
  echo "$DSUB_OUT" >&2
  echo "dsub failed; see output above" >&2
  exit 1
fi
echo "job_id=$JOB_ID"
echo "poll: djob -ll $JOB_ID"
