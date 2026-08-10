#!/bin/bash
#SBATCH --job-name=recap_setup
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
ROOT=${1:?root required}
VENV=${2:?venv required}
DATA_ROOT=${3:?data root required}
source "$VENV/bin/activate"
export RECAP_ROOT="$ROOT"
export RECAP_DATA_ROOT="$DATA_ROOT"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/recap_matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd "$ROOT"
python -m pytest
python -m src.smoke --data-root "$DATA_ROOT"
