#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -z "${SCRATCH:-}" ]]; then
  echo "SCRATCH is not set; run this on a Narval login node." >&2
  exit 2
fi
VENV=${RECAP_VENV:-"$SCRATCH/recap_pilots_venv_py311"}
CACHE_ROOT=${RECAP_CACHE_ROOT:-"$SCRATCH/recap_pilots_cache"}
DATA_ROOT=${RECAP_DATA_ROOT:-"$SCRATCH/recap_mnist"}
ACCOUNT=${RECAP_ACCOUNT:-${SLURM_ACCOUNT:-}}
SETUP_TIME_LIMIT=${RECAP_TIME_LIMIT:-${RECAP_SETUP_TIME_LIMIT:-02:00:00}}
CALIBRATE_TIME_LIMIT=${RECAP_TIME_LIMIT:-${RECAP_CALIBRATE_TIME_LIMIT:-01:00:00}}
RUN_TIME_LIMIT=${RECAP_TIME_LIMIT:-${RECAP_RUN_TIME_LIMIT:-06:00:00}}
AGGREGATE_TIME_LIMIT=${RECAP_TIME_LIMIT:-${RECAP_AGGREGATE_TIME_LIMIT:-02:00:00}}
RESOLVE_TIME_LIMIT=${RECAP_TIME_LIMIT:-${RECAP_RESOLVE_TIME_LIMIT:-00:30:00}}
REPORT_TIME_LIMIT=${RECAP_TIME_LIMIT:-${RECAP_REPORT_TIME_LIMIT:-01:00:00}}
export RECAP_CACHE_ROOT="$CACHE_ROOT"
cd "$ROOT"
if ! mkdir -p "$ROOT/logs" "$ROOT/results/pilots" "$DATA_ROOT" "$CACHE_ROOT"; then
  echo "Unable to create launch directories. Your current filesystem quota may still be exhausted." >&2
  echo "Run diskusage_report and inspect ~/.local/share/virtualenv and ~/.cache before retrying." >&2
  exit 2
fi

# Keep the Alliance Arrow/Python module environment in this shell so it is also
# exported to submitted jobs. Each job loads it again defensively.
source "$ROOT/scripts/load_modules.sh"
"$ROOT/scripts/bootstrap_env.sh" "$ROOT" "$VENV"
source "$VENV/bin/activate"
export RECAP_ROOT="$ROOT"
python -m src.manifest generate
# Narval compute nodes do not have general internet access. Download once on the login node;
# this stages data only and performs no training.
python -m src.stage_data --data-root "$DATA_ROOT"

account_args=()
if [[ -n "$ACCOUNT" ]]; then
  account_args=(--account="$ACCOUNT")
else
  echo "RECAP_ACCOUNT is unset; sbatch will use your default Slurm account." >&2
fi

submit() {
  local time_limit=${1:?time limit required}
  shift
  sbatch --parsable "${account_args[@]}" --time="$time_limit" "$@" | cut -d';' -f1
}

setup=$(submit "$SETUP_TIME_LIMIT" "$ROOT/slurm/setup.sbatch" "$ROOT" "$VENV" "$DATA_ROOT")
cpu_cal=$(submit "$CALIBRATE_TIME_LIMIT" --dependency="afterok:$setup" "$ROOT/slurm/calibrate.sbatch" "$ROOT" "$VENV" "$DATA_ROOT" cpu)
if [[ "${RECAP_SKIP_GPU_CALIBRATION:-0}" == "1" ]]; then
  gpu_cal="skipped"
else
  gpu_cal=$(submit "$CALIBRATE_TIME_LIMIT" --dependency="afterok:$setup" --gres=gpu:1 "$ROOT/slurm/calibrate.sbatch" "$ROOT" "$VENV" "$DATA_ROOT" cuda)
fi

a=$(submit "$RUN_TIME_LIMIT" --dependency="afterok:$cpu_cal" --array=0-23%12 --job-name=recap_A "$ROOT/slurm/run_array.sbatch" "$ROOT" "$VENV" "$DATA_ROOT" A)
b=$(submit "$RUN_TIME_LIMIT" --dependency="afterok:$cpu_cal" --array=0-7%8 --job-name=recap_B "$ROOT/slurm/run_array.sbatch" "$ROOT" "$VENV" "$DATA_ROOT" B)
a_agg=$(submit "$AGGREGATE_TIME_LIMIT" --dependency="afterok:$a" --job-name=recap_A_agg "$ROOT/slurm/aggregate.sbatch" "$ROOT" "$VENV" A)
b_agg=$(submit "$AGGREGATE_TIME_LIMIT" --dependency="afterok:$b" --job-name=recap_B_agg "$ROOT/slurm/aggregate.sbatch" "$ROOT" "$VENV" B)
a_retry=$(submit "$RUN_TIME_LIMIT" --dependency="afterok:$a_agg" --array=0 --job-name=recap_A_retry "$ROOT/slurm/run_array.sbatch" "$ROOT" "$VENV" "$DATA_ROOT" AR)
b_retry=$(submit "$RUN_TIME_LIMIT" --dependency="afterok:$a_agg:$b_agg" --array=0-3%4 --job-name=recap_B_retry "$ROOT/slurm/run_array.sbatch" "$ROOT" "$VENV" "$DATA_ROOT" BR)
a_final=$(submit "$AGGREGATE_TIME_LIMIT" --dependency="afterok:$a_retry" --job-name=recap_A_final "$ROOT/slurm/aggregate.sbatch" "$ROOT" "$VENV" A finalize)
b_final=$(submit "$AGGREGATE_TIME_LIMIT" --dependency="afterok:$b_retry" --job-name=recap_B_final "$ROOT/slurm/aggregate.sbatch" "$ROOT" "$VENV" B finalize)
resolve=$(submit "$RESOLVE_TIME_LIMIT" --dependency="afterok:$a_final:$b_final" "$ROOT/slurm/resolve.sbatch" "$ROOT" "$VENV" resolve)

prep=$(submit "$RUN_TIME_LIMIT" --dependency="afterok:$resolve" --array=0-2%3 --job-name=recap_prepare "$ROOT/slurm/run_array.sbatch" "$ROOT" "$VENV" "$DATA_ROOT" P)
c=$(submit "$RUN_TIME_LIMIT" --dependency="afterok:$prep" --array=0-1%2 --job-name=recap_C "$ROOT/slurm/run_array.sbatch" "$ROOT" "$VENV" "$DATA_ROOT" C)
c_agg=$(submit "$AGGREGATE_TIME_LIMIT" --dependency="afterok:$c" --job-name=recap_C_agg "$ROOT/slurm/aggregate.sbatch" "$ROOT" "$VENV" C)
d=$(submit "$RUN_TIME_LIMIT" --dependency="afterok:$c_agg" --array=0-11%12 --job-name=recap_D "$ROOT/slurm/run_array.sbatch" "$ROOT" "$VENV" "$DATA_ROOT" D)
d_agg=$(submit "$AGGREGATE_TIME_LIMIT" --dependency="afterok:$d" --job-name=recap_D_agg "$ROOT/slurm/aggregate.sbatch" "$ROOT" "$VENV" D)
d_retry=$(submit "$RUN_TIME_LIMIT" --dependency="afterok:$d_agg" --array=0-5%6 --job-name=recap_D_retry "$ROOT/slurm/run_array.sbatch" "$ROOT" "$VENV" "$DATA_ROOT" DR)
d_final=$(submit "$AGGREGATE_TIME_LIMIT" --dependency="afterok:$d_retry" --job-name=recap_D_final "$ROOT/slurm/aggregate.sbatch" "$ROOT" "$VENV" D finalize)
resolve_e=$(submit "$RESOLVE_TIME_LIMIT" --dependency="afterok:$d_final" "$ROOT/slurm/resolve.sbatch" "$ROOT" "$VENV" resolve-e)
e=$(submit "$RUN_TIME_LIMIT" --dependency="afterok:$resolve_e" --array=0-11%12 --job-name=recap_E "$ROOT/slurm/run_array.sbatch" "$ROOT" "$VENV" "$DATA_ROOT" E)
e_agg=$(submit "$AGGREGATE_TIME_LIMIT" --dependency="afterok:$e" --job-name=recap_E_agg "$ROOT/slurm/aggregate.sbatch" "$ROOT" "$VENV" E)
report=$(submit "$REPORT_TIME_LIMIT" --dependency="afterok:$e_agg" "$ROOT/slurm/report.sbatch" "$ROOT" "$VENV")

python - "$ROOT/results/pilots/submission.json" "$setup" "$cpu_cal" "$gpu_cal" "$a" "$b" "$a_agg" "$b_agg" "$a_retry" "$b_retry" "$a_final" "$b_final" "$resolve" "$prep" "$c" "$c_agg" "$d" "$d_agg" "$d_retry" "$d_final" "$resolve_e" "$e" "$e_agg" "$report" <<'PY'
import json, sys
keys = ["setup", "cpu_calibration", "gpu_calibration", "A", "B", "A_aggregate", "B_aggregate", "A_retry", "B_retry", "A_finalize", "B_finalize", "resolve", "checkpoint_prep", "C", "C_aggregate", "D", "D_aggregate", "D_retry", "D_finalize", "resolve_E", "E", "E_aggregate", "report"]
with open(sys.argv[1], "w") as handle:
    json.dump(dict(zip(keys, sys.argv[2:])), handle, indent=2)
    handle.write("\n")
PY

echo "Submitted ReCAP pilot DAG. Final report job: $report"
echo "Job registry: $ROOT/results/pilots/submission.json"
