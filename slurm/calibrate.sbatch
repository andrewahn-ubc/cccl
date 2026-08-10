#!/bin/bash
#SBATCH --job-name=recap_calibrate
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
ROOT=${1:?root required}
VENV=${2:?venv required}
DATA_ROOT=${3:?data root required}
DEVICE=${4:?device required}
if [[ "$DEVICE" == "cuda" && -n "${RECAP_CUDA_MODULE:-}" ]]; then
  module load "$RECAP_CUDA_MODULE"
fi
source "$VENV/bin/activate"
export RECAP_ROOT="$ROOT"
export RECAP_DATA_ROOT="$DATA_ROOT"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/recap_matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd "$ROOT"
python -m src.calibrate --device "$DEVICE" --steps 50
