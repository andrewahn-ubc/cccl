#!/bin/bash
#SBATCH --job-name=recap_resolve
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
ROOT=${1:?root required}
VENV=${2:?venv required}
COMMAND=${3:-resolve}
source "$ROOT/scripts/load_modules.sh"
source "$VENV/bin/activate"
export RECAP_ROOT="$ROOT"
cd "$ROOT"
python -m src.manifest "$COMMAND"
