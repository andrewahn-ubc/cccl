#!/bin/bash
#SBATCH --job-name=recap_setup
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
ROOT=${1:?root required}
VENV=${2:?venv required}
DATA_ROOT=${3:?data root required}
source "$ROOT/scripts/load_modules.sh"
source "$VENV/bin/activate"
export RECAP_ROOT="$ROOT"
export RECAP_DATA_ROOT="$DATA_ROOT"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/recap_matplotlib"
mkdir -p "$MPLCONFIGDIR"
cd "$ROOT"

stage() {
  echo "[$(date -Is)] $*"
}

stage "checking scientific Python imports"
python - <<'PY'
import matplotlib
import pandas
import pyarrow
import scipy
import torch
import torchvision
import yaml

print(
    "imports OK:",
    f"torch={torch.__version__}",
    f"torchvision={torchvision.__version__}",
    f"pandas={pandas.__version__}",
    f"pyarrow={pyarrow.__version__}",
    f"scipy={scipy.__version__}",
    f"matplotlib={matplotlib.__version__}",
    f"pyyaml={yaml.__version__}",
)
PY

if [[ "${RECAP_RUN_CLUSTER_TESTS:-0}" == "1" ]]; then
  stage "running opt-in unit-test suite"
  python -m pytest -vv --durations=20
else
  stage "unit-test suite skipped (set RECAP_RUN_CLUSTER_TESTS=1 to enable)"
fi

stage "running MNIST training smoke test"
python -m src.smoke --data-root "$DATA_ROOT"
stage "setup completed"
