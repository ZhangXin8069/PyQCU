#!/bin/bash
# Submit ONE formal `pyqcu.test --l48-two-node` job to the LX2 scheduler.
#
# Test-only helper (`pyqcu/testing/**` is not shipped): it drives the formal
# port, it does not implement any test logic.  The direction of the dependency
# matters - the job script calls `python3 -m pyqcu.test.launch`; nothing under
# `pyqcu/test` may import `pyqcu.testing`.
#
# Why this script looks the way it does (2026-11-17 remote session):
#
#   * The verified two-node route is the CCS `drun` path, not ssh: job 2155536
#     (cn[19544,19546], 32 ranks, c64, PASS) ran
#       export CCS_DRUN_RANK_FILE=$OUT/logs/$JOBTAG-rankfile
#       /opt/batch/agent/tools/drun -mpi pmix -nn 2 -rpn 16 -R core=18 -v -- ...
#     with the rankfile derived from `CCS_ALLOC_FILE`
#     (`host total_cores rank_count`, 288 cores / 16 ranks = 18 cores per rank).
#     See docs/pyqcu_test_l48_remote.md sec. 0.1.6 and
#     docs/kupl_remote_l48_experience_20261117.md sec. 7.5.
#   * `drun` warns that -N/-nn/-rpn/-nl/-R are ignored once the rankfile
#     variable is set.  The rankfile is the placement, so a missing or short
#     rankfile silently shrinks the launch: this script writes it through
#     `pyqcu.test.launch` (which validates 32 rows, 18 disjoint cores each,
#     prod(grid)==ranks and ranks == nodes*ranks_per_node) and never infers the
#     rank count from the drun flags.  Check the report's
#     mpi_size/grid/local_lattice instead.
#   * Compute-node ssh is NOT a usable launcher for this account: the key is
#     accepted ("Server accepts key") and the session is then closed.  Do not
#     restore `plm_rsh`/`--mca plm donau`; `--launcher mpirun` exists only for
#     a site where a PLM does work.
#   * `drun` exists only on the compute nodes, so every launcher decision is
#     made inside the job body, not here on the login node.
#   * Serial, single job: one dsub allocation, one task, one 32-rank launch.
#     The rows run one after another inside it.  Do not submit sharded copies -
#     two exclusive 2-node requests cannot be placed at the same time.
#   * The rank grid is the L48 production shape `2 2 2 4` (48^3 x 96, 24^3 x 24
#     per rank).  `pyqcu.test.launch` re-checks prod(grid)==ranks and the even
#     rank-local extents, so a `--ranks` that does not fit is refused instead of
#     being launched at the wrong size.
#   * This script never decides PASS/FAIL and runs nothing locally: without
#     --dry-run it only writes the job script and calls `dsub -s`.
#
# Usage:
#   ./submit_pyqcu_test.sh --dry-run                       # review the script
#   ./submit_pyqcu_test.sh --tag l48_2n32r_formal_001      # submit one job
#   ./submit_pyqcu_test.sh --only kupl/l48/bicgstab/c64 --threads 18
#
# Poll with `bash -lc 'djob -ll <job_id>'` and require exactly two execNodes
# plus a report whose mpi_size is 32; a PENDING job carries no numerical
# information.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DSUB="${DSUB:-/opt/batch/cli/bin/dsub}"
QUEUE="${QUEUE:-q_hpcapp}"
OUTDIR="${OUTDIR:-$REPO/data/pyqcu_test_l48_20261118}"
HPCKIT="${HPCKIT:-/work_ssd/software/HPCKit/26.1.RC1}"
DRUN="${DRUN:-/opt/batch/agent/tools/drun}"

TAG=""
NODES=2
RANKS_PER_NODE=16
RANKS=32
THREADS=18
CORES_PER_RANK=18
ONLY="kupl/l48"
WALLTIME=1800
NODE=""
LAUNCHER="drun"
EXTRA=""
DRY_RUN=0
EXCLUSIVE=1

usage() {
  cat <<'EOF'
Usage: submit_pyqcu_test.sh [options]

  --tag TAG             job/tag name (default: l48test_<pid>)
  --nodes N             compute nodes (default 2)
  --ranks-per-node N    ranks per node (default 16)
  --ranks N             total ranks (default 32; must equal nodes*ranks-per-node)
  --threads N           OpenMP threads per rank (default 18)
  --cores-per-rank N    physical cores per rank in the rankfile (default 18)
  --only PATTERN        --only value of the formal command (default kupl/l48)
  --node NAME           optional scheduler node preference (-nl)
  --walltime SEC        scheduler wall time (default 1800)
  --outdir DIR          output root (default data/pyqcu_test_l48_20261118)
  --hpckit DIR          HPCKit root exported inside the job
                        (default /work_ssd/software/HPCKit/26.1.RC1)
  --drun PATH           drun executable used by the job
                        (default /opt/batch/agent/tools/drun)
  --launcher drun|mpirun  rankfile launcher (default drun: the verified route;
                        mpirun needs a working PLM, which this account has not)
  --shared              omit -x job (allow a partially used node)
  --extra "ARGS"        appended to the rendered formal pyqcu.test command
  --dry-run             write the job script and print it, do not submit
  -h, --help            this help

The job writes <outdir>/logs/<tag>-rankfile from $CCS_ALLOC_FILE, exports it
as CCS_DRUN_RANK_FILE and launches the formal port.  It refuses to start when
the allocation is missing or the rankfile cannot be validated.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --nodes) NODES="$2"; shift 2 ;;
    --ranks-per-node) RANKS_PER_NODE="$2"; shift 2 ;;
    --ranks) RANKS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --cores-per-rank) CORES_PER_RANK="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --node) NODE="$2"; shift 2 ;;
    --walltime) WALLTIME="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --hpckit) HPCKIT="$2"; shift 2 ;;
    --drun) DRUN="$2"; shift 2 ;;
    --launcher) LAUNCHER="$2"; shift 2 ;;
    --shared) EXCLUSIVE=0; shift ;;
    --extra) EXTRA="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for value in NODES RANKS_PER_NODE RANKS THREADS CORES_PER_RANK WALLTIME; do
  if ! [[ "${!value}" =~ ^[0-9]+$ ]]; then
    echo "--${value,,} must be an integer, got '${!value}'" >&2
    exit 2
  fi
done
if [ -z "$TAG" ]; then
  TAG="l48test_$$"
fi
if [ "$LAUNCHER" != "drun" ] && [ "$LAUNCHER" != "mpirun" ]; then
  echo "--launcher must be drun or mpirun, got '$LAUNCHER'" >&2
  exit 2
fi
if [ -z "$ONLY" ]; then
  echo "--only must not be empty" >&2
  exit 2
fi
if [ $((NODES * RANKS_PER_NODE)) -ne "$RANKS" ]; then
  echo "ranks=$RANKS != nodes=$NODES * ranks-per-node=$RANKS_PER_NODE" >&2
  exit 2
fi
if [ $((RANKS_PER_NODE * THREADS)) -gt 608 ]; then
  echo "ranks-per-node*threads=$((RANKS_PER_NODE * THREADS)) exceeds the 608" \
       "cores of one node; use a smaller pair" >&2
  exit 2
fi
if [ $((RANKS_PER_NODE * CORES_PER_RANK)) -gt 608 ]; then
  echo "ranks-per-node*cores-per-rank=$((RANKS_PER_NODE * CORES_PER_RANK))" \
       "exceeds the 608 cores of one node" >&2
  exit 2
fi

EXTRA_ARR=()
if [ -n "$EXTRA" ]; then
  # shellcheck disable=SC2206
  read -r -a EXTRA_ARR <<< "$EXTRA"
fi
EXTRA_QUOTED=""
if [ "${#EXTRA_ARR[@]}" -gt 0 ]; then
  EXTRA_QUOTED="$(printf '%q ' "${EXTRA_ARR[@]}")"
fi

mkdir -p "$OUTDIR"/{logs,jobs,rows,reports,solutions,caches}

JOB="$OUTDIR/jobs/$TAG.sh"
{
  echo '#!/bin/bash'
  echo "#DSUB -q $QUEUE"
  echo "#DSUB -n $TAG"
  echo "#DSUB -nn $NODES"
  echo "#DSUB -N $RANKS"
  echo "#DSUB -rpn $RANKS_PER_NODE"
  echo "#DSUB -R \"cpu=$THREADS;mem=8G\""
  # Exclusive by default: a 2x16 L48 rank set wants two whole nodes.
  if [ "$EXCLUSIVE" = "1" ]; then echo '#DSUB -x job'; fi
  echo "#DSUB -oo $OUTDIR/logs/$TAG.out"
  echo "#DSUB -eo $OUTDIR/logs/$TAG.err"
  echo "#DSUB -T $WALLTIME"
  if [ -n "$NODE" ]; then echo "#DSUB -nl $NODE"; fi
  echo 'set +u'
  echo "source $HPCKIT/setvars.sh >/dev/null 2>&1 || true"
  echo 'set -u'
  echo "export PATH=$HPCKIT/hmpi/bisheng/release/hmpi/bin:/work_ssd/software/soft/tool/anaconda3/2023.3/bin:/opt/batch/cli/bin:\$PATH"
  echo "export LD_LIBRARY_PATH=$HPCKIT/hmpi/bisheng/release/hmpi/lib:$HPCKIT/compiler/gcc/lib64:$HPCKIT/compiler/bisheng/lib:/work_ssd/software/HPCKit/alllib:\${LD_LIBRARY_PATH-}"
  echo "export OPAL_PREFIX=$HPCKIT/hmpi/bisheng/release/hmpi"
  echo "export PYTHONPATH=$REPO:\${PYTHONPATH-}"
  echo "export OMP_NUM_THREADS=$THREADS KUPL_OMP_THREADS=$THREADS"
  echo 'export KUPL_SHM_HALO=0 OMPI_MCA_rmaps_base_oversubscribe=1'
  echo "cd $REPO"
  echo ''
  echo "TAG=$TAG"
  echo "JOBID=\${CCS_JOB_ID:-$TAG}"
  echo "RANKFILE=\"$OUTDIR/logs/\${TAG}-rankfile\""
  echo "ARGVFILE=\"$OUTDIR/logs/\${TAG}-argv\""
  echo '# The rankfile selects the CCS launcher; drun ignores -nn/-rpn/-R then.'
  echo 'export CCS_DRUN_RANK_FILE="$RANKFILE"'
  echo 'if [ -z "${CCS_ALLOC_FILE-}" ] || [ ! -f "${CCS_ALLOC_FILE}" ]; then'
  echo '  echo "[launch] no CCS_ALLOC_FILE in the job: refusing the cross-node launch" >&2'
  echo '  exit 2'
  echo 'fi'
  echo 'echo "[launch] alloc_file=$CCS_ALLOC_FILE"'
  echo 'echo "[launch] rankfile=$RANKFILE job=$JOBID"'
  echo "# Validate \$CCS_ALLOC_FILE and write the rankfile ($RANKS rows, $CORES_PER_RANK cores each)."
  echo "python3 -B -m pyqcu.test.launch rankfile --alloc-file \"\$CCS_ALLOC_FILE\" \\"
  echo "  --rankfile \"\$RANKFILE\" --job-id \"\$JOBID\" \\"
  echo "  --nodes $NODES --ranks-per-node $RANKS_PER_NODE --ranks $RANKS \\"
  echo "  --cores-per-rank $CORES_PER_RANK --grid 2 2 2 4 --lattice 48 48 48 96 \\"
  echo '  || { echo "[launch] rankfile refused; not launching" >&2; exit 2; }'
  echo '# Render the launcher argv: one argument per line, no eval.'
  echo "python3 -B -m pyqcu.test.launch argv --launcher $LAUNCHER \\"
  echo "  --drun '$DRUN' \\"
  echo "  --alloc-file \"\$CCS_ALLOC_FILE\" --rankfile \"\$RANKFILE\" --job-id \"\$JOBID\" \\"
  echo "  --nodes $NODES --ranks-per-node $RANKS_PER_NODE --ranks $RANKS \\"
  echo "  --cores-per-rank $CORES_PER_RANK --grid 2 2 2 4 --lattice 48 48 48 96 \\"
  echo "  --only '$ONLY' --threads $THREADS --outdir '$OUTDIR' \\"
  echo "  > \"\$ARGVFILE\" || { echo \"[launch] plan refused; not launching\" >&2; exit 2; }"
  echo 'mapfile -t LAUNCH_ARGV < "$ARGVFILE"'
  if [ -n "$EXTRA_QUOTED" ]; then
    echo "EXTRA_ARGS=($EXTRA_QUOTED)"
    echo 'LAUNCH_ARGV+=("${EXTRA_ARGS[@]}")'
  fi
  echo 'if [ "${#LAUNCH_ARGV[@]}" -lt 2 ]; then'
  echo '  echo "[launch] empty launch argv" >&2'
  echo '  exit 2'
  echo 'fi'
  echo 'echo "[launch] rows=$(wc -l < "$RANKFILE") argv=$ARGVFILE"'
  echo 'echo "[launch] ${LAUNCH_ARGV[*]}"'
  echo '"${LAUNCH_ARGV[@]}"'
  echo 'RC=$?'
  echo 'echo "[launch] rc=$RC"'
  echo 'exit $RC'
} > "$JOB"
chmod +x "$JOB"

echo "job_script=$JOB"
echo "tag=$TAG nodes=$NODES ranks=$RANKS rpn=$RANKS_PER_NODE threads=$THREADS"
echo "launcher=$LAUNCHER only=$ONLY node=${NODE:-auto} walltime=$WALLTIME"
echo "formal: python3 -B -m pyqcu.test --mode solve --l48-two-node" \
     "--only '$ONLY' --ranks $RANKS --lattice-grid 2 2 2 4 --threads $THREADS"
echo "(the exact launcher argv is rendered inside the job: see the job script)"

if [ "$DRY_RUN" = "1" ]; then
  echo "--- $JOB ---"
  cat "$JOB"
  echo "--- end $JOB ---"
  echo "dry-run: would run 'dsub -s $JOB'"
  exit 0
fi

if [ ! -x "$DSUB" ]; then
  echo "dsub not found at $DSUB; run on a login node or set DSUB=" >&2
  exit 2
fi

# `#DSUB` blocks are only parsed with -s/--script, and the batch CLI needs a
# login shell for its environment; poll with the same shell.
DSUB_OUT="$(bash -lc "dsub -s '$JOB'" 2>&1)"
JOB_ID="$(printf '%s\n' "$DSUB_OUT" | grep -o '[0-9][0-9]*' | head -1)"
if [ -z "$JOB_ID" ]; then
  echo "$DSUB_OUT" >&2
  echo "dsub failed; see output above" >&2
  exit 1
fi
echo "job_id=$JOB_ID"
echo "poll: bash -lc 'djob -ll $JOB_ID'   # require two execNodes; PENDING is not a result"
