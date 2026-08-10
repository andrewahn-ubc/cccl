#!/bin/bash
#SBATCH --job-name=recap_run
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
ROOT=${1:?root required}
VENV=${2:?venv required}
DATA_ROOT=${3:?data root required}
PILOT=${4:?pilot required}
source "$ROOT/scripts/load_modules.sh"
source "$VENV/bin/activate"
export RECAP_ROOT="$ROOT"
export RECAP_DATA_ROOT="$DATA_ROOT"
export RECAP_DEVICE=cpu
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/recap_matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd "$ROOT"

child=""
forward_signal() {
  if [[ -n "$child" ]]; then
    kill -USR1 "$child" 2>/dev/null || true
  fi
}
trap forward_signal USR1 TERM
set +e
python -m src.pilots --manifest "$ROOT/manifests/pilot_${PILOT}.csv" --row "$SLURM_ARRAY_TASK_ID" &
child=$!
while true; do
  wait "$child"
  status=$?
  if kill -0 "$child" 2>/dev/null; then
    continue
  fi
  break
done
set -e
if [[ $status -eq 99 ]]; then
  target=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
  scontrol requeue "$target"
  exit 0
fi
exit "$status"
