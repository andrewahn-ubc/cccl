# ReCAP 12-hour go/no-go pilots

This repository implements the complete recurring Permuted-MNIST pilot suite in [`RECAP_12H_GO_NO_GO_PILOTS.md`](RECAP_12H_GO_NO_GO_PILOTS.md): deterministic paired task libraries, Pilots A–E, shared C/D checkpoints, scientific gates, plots, Slurm arrays, resume-safe checkpoints, and an automatically generated decision report.

Before the launch, verify the configurable module/resource defaults against the current Alliance pages for [Narval](https://docs.alliancecan.ca/wiki/Narval), [Python](https://docs.alliancecan.ca/wiki/Python), [running jobs](https://docs.alliancecan.ca/wiki/Running_jobs), and [storage](https://docs.alliancecan.ca/wiki/Storage_and_file_management).

## Narval: submit the entire pilot DAG

On a Narval login node:

```bash
cd "$PROJECT"
git clone <YOUR-REPOSITORY-URL> recap
cd recap
RECAP_ACCOUNT=def-youraccount bash scripts/submit_all.sh
```

That command performs login-safe setup only: it loads the Alliance Python and Arrow modules, creates the virtual environment at `$SCRATCH/recap_pilots_venv_py311`, puts virtualenv/pip/temp caches under `$SCRATCH/recap_pilots_cache`, generates deterministic manifests/task libraries, and downloads MNIST once into `$SCRATCH/recap_mnist`. It then submits all training, aggregation, and reporting jobs. It never trains on the login node. Keeping these high-file-count paths off `$HOME` avoids Alliance home quota failures.

If your Slurm association has a working default account, `RECAP_ACCOUNT` may be omitted. Useful overrides are:

```bash
RECAP_PYTHON_MODULE=3.11 \
RECAP_ARROW_MODULE=arrow \
RECAP_VENV="$SCRATCH/recap_pilots_venv_py311" \
RECAP_CACHE_ROOT="$SCRATCH/recap_pilots_cache" \
RECAP_DATA_ROOT="$SCRATCH/recap_mnist" \
RECAP_CUDA_MODULE=cuda/12.2 \
bash scripts/submit_all.sh
```

Time limits include headroom for Narval's shared-filesystem I/O: setup 3 hours, calibration 1.5 hours, each training array element 9 hours, aggregation 3 hours, manifest resolution 45 minutes, and reporting 1.5 hours. Override a class with `RECAP_SETUP_TIME_LIMIT`, `RECAP_CALIBRATE_TIME_LIMIT`, `RECAP_RUN_TIME_LIMIT`, `RECAP_AGGREGATE_TIME_LIMIT`, `RECAP_RESOLVE_TIME_LIMIT`, or `RECAP_REPORT_TIME_LIMIT`. `RECAP_TIME_LIMIT` remains an optional global override.

`RECAP_ARROW_MODULE` defaults to the site's default compatible Arrow module; override it with a version reported by `module spider arrow` only when necessary. Alliance provides PyArrow through that module and intentionally blocks wheelhouse installation without it. `RECAP_CUDA_MODULE` is optional and should be set only if the current Narval PyTorch wheel requires a site CUDA module. Check `module spider cuda` rather than copying the example version blindly. Set `RECAP_SKIP_GPU_CALIBRATION=1` if the account cannot request GPUs.

The setup job checks all compiled scientific-Python imports and runs a real three-update MNIST training/checkpoint smoke test. The full unit-test suite is intentionally not repeated on Narval for every submission; set `RECAP_RUN_CLUSTER_TESTS=1` to opt in. Compute jobs put temporary files, bytecode, and XDG caches on node-local `$SLURM_TMPDIR` rather than the shared filesystem.

### Recovering from a home-quota failure

An older launcher could make `virtualenv` build pip/setuptools seed images in `$HOME`. If that attempt already filled the quota, inspect it before rerunning:

```bash
diskusage_report
du -sh ~/.local/share/virtualenv ~/.cache/pip ~/.cache/virtualenv 2>/dev/null
```

Those directories contain regenerable installer caches, not the project environment. After confirming the paths, the safest recovery is to move them to scratch:

```bash
mkdir -p "$SCRATCH/recap_quota_recovery"
test ! -e ~/.local/share/virtualenv || mv ~/.local/share/virtualenv "$SCRATCH/recap_quota_recovery/virtualenv-app-data"
test ! -e ~/.cache/pip || mv ~/.cache/pip "$SCRATCH/recap_quota_recovery/pip-cache"
test ! -e ~/.cache/virtualenv || mv ~/.cache/virtualenv "$SCRATCH/recap_quota_recovery/virtualenv-cache"
```

Then pull this fix and rerun the same `RECAP_ACCOUNT=... bash scripts/submit_all.sh` command. A partial old `recap/.venv` is no longer used because the default environment now lives on scratch.

Set `RECAP_ALLOW_INDEX=1` only on systems without the Alliance wheelhouse; this allows ordinary package-index installation. Module versions and the account remain configuration, never hard-coded scientific inputs.

The dependency graph is:

```text
setup/smoke ─ CPU calibration ─┬─ A[24] ─ A aggregate ─┐
                              └─ B[8]  ─ B aggregate ─┴─ conditional A/B diagnostic ─ resolve width
setup/smoke ─ GPU calibration (timing only)                       │
                                                                 ▼
                    shared ER checkpoints[3] ─ C[2] ─ C aggregate
                                                        │
                                                        ▼
                                                   D[12] ─ aggregate ─ conditional D repair
                                                        │
                                                        ▼
                                                   E[12] ─ aggregate ─ report
```

A, B, and all within-pilot cells run as capped arrays. The width-200 A diagnostic, stronger-skew B retry, and gentler 10%/100-step D repair are pre-submitted but exit instantly unless their exact gate triggers them. Downstream rows likewise write explicit `skipped` artifacts when an upstream scientific gate says to stop. Pilot E therefore consumes no training budget after a decisive D failure. C adopts Fisher or activation utility if its predeclared fallback gate passes, and uses empirical ablation for the metric-independent upper-bound diagnostic if all cheap metrics fail.

The final output is:

```text
results/pilots/PILOT_DECISION_REPORT.md
```

Each run lives at `results/pilots/<pilot>/<run_id>/` and contains `config.yaml`, `metrics.parquet`, `events.jsonl`, `checkpoint.pt`, `status.json`, `environment.txt`, and `DONE`. Pilot-level directories contain the required CSV, JSON, and PNG artifacts. `results/pilots/submission.json` records every submitted job ID.

## Failure and preemption handling

Array jobs request a two-minute pre-timeout signal. Online runs checkpoint atomically at block boundaries; a signalled task requeues itself and resumes from the last complete block. Completed rows are idempotent. To resubmit only missing/failed rows with their identical configs:

```bash
bash scripts/resubmit_missing.sh E
```

Scientific marginal-case retries are already part of the DAG but run only when triggered by a predeclared gate. Software-invalid rows are rerun unchanged with the helper above.

## Local correctness checks

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m pytest
```

The tests cover transition/occupancy math, masks and Taylor-score purity, function-neutral reset, optimizer-state slice clearing, replay quotas/capacity/sampling, pre-update metric order, and deterministic checkpoint resume.

## Protocol notes

- The first ten blocks contain a seed-fixed permutation of all ten tasks; the remaining blocks use the exact `.30,.30,.10,.10,.10,.02×5` generator.
- Online updates keep the total optimization batch at 64 (current/replay split 32/32 once replay exists). Offline uses 64 mixture examples.
- The common replay policy reserves 50 slots per task. Pilot B’s oracle policy dynamically applies largest-remainder quotas with a five-example minimum for observed tasks.
- CPU is the default for MNIST arrays. CPU and GPU calibrations are both submitted, but GPU queue latency cannot stall the scientific DAG.
- All gates use the frozen thresholds in the protocol; aggregators never tune thresholds from observed results.
