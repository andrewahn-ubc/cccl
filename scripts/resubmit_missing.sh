#!/bin/bash
set -euo pipefail

PILOT=${1:?usage: scripts/resubmit_missing.sh A|AR|B|BR|P|C|D|DR|E}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${RECAP_DATA_ROOT:-"${SCRATCH:?SCRATCH must be set}/recap_mnist"}
VENV=${RECAP_VENV:-"$SCRATCH/recap_pilots_venv_py311"}
ACCOUNT=${RECAP_ACCOUNT:-${SLURM_ACCOUNT:-}}
source "$VENV/bin/activate"
indices=$(python -m src.missing --pilot "$PILOT")
if [[ -z "$indices" ]]; then
  echo "No missing or failed rows for Pilot $PILOT."
  exit 0
fi
account_args=()
if [[ -n "$ACCOUNT" ]]; then account_args=(--account="$ACCOUNT"); fi
job=$(sbatch --parsable "${account_args[@]}" --array="$indices%12" --job-name="recap_${PILOT}_retry" "$ROOT/slurm/run_array.sbatch" "$ROOT" "$VENV" "$DATA_ROOT" "$PILOT" | cut -d';' -f1)
echo "Resubmitted Pilot $PILOT rows $indices as job $job (identical configs)."
