#!/bin/bash
#SBATCH --job-name=recap_aggregate
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
ROOT=${1:?root required}
VENV=${2:?venv required}
PILOT=${3:?pilot required}
MODE=${4:-preliminary}
source "$ROOT/scripts/load_modules.sh"
source "$VENV/bin/activate"
export RECAP_ROOT="$ROOT"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/recap_matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd "$ROOT"
args=(--pilot "$PILOT")
if [[ "$MODE" == "finalize" ]]; then args+=(--finalize); fi
python -m src.aggregate "${args[@]}"
