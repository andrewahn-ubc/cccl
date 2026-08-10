#!/bin/bash
#SBATCH --job-name=recap_report
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
ROOT=${1:?root required}
VENV=${2:?venv required}
source "$ROOT/scripts/load_modules.sh"
source "$VENV/bin/activate"
export RECAP_ROOT="$ROOT"
cd "$ROOT"
python -m src.report
