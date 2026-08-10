# ReCAP Preliminary Go/No-Go Pilots (Narval, ≤12 Hours)

## 0. Purpose and hard constraints

This plan answers five questions in order:

1. Can the architecture represent the relevant mixture?
2. Does unequal future demand create exploitable value?
3. Does the Taylor score predict deletion damage?
4. Does consolidation recover behavior after compression?
5. With perfect future-demand information, does end-to-end ReCAP beat ER and random recycling?

Question 5 is the principal kill test. The goal is evidence, not polished benchmarking. Use recurring Permuted MNIST only, one development task library, aggressive early stopping, and paired seeds. Target 8–10 hours elapsed, leaving two hours for queue variation/retries. If Narval wait time dominates, run MNIST pilots on CPU nodes; profile one CPU and one GPU smoke run first.

## 1. Common implementation

### Data and stream

- 10 Permuted-MNIST tasks; independent fixed input and label permutations.
- Append task one-hot; shared 10-class head.
- Two-layer ReLU MLP; batch 64; Adam `lr=1e-3` unless smoke tuning shows divergence.
- Fixed total replay/probe buffer `B=500`, equal allocation (50/task), reservoir replacement within task.
- Main skew probabilities: tasks 0–1 each `.30`, tasks 2–4 each `.10`, tasks 5–9 each `.02` (sums to 1).
- End-to-end stream: 30 blocks × 150 updates; first appearance of each task is forced once, then sample from the skew distribution.
- For every metric, predict before updating. Save task permutations and schedule once and reuse across methods.

### ReCAP pilot defaults

- Oracle future occupancy from the true current generator; `H=10`, `gamma=.9`.
- Gate-Taylor distortion, layer-median normalization.
- Recycle lowest 20% per hidden layer every boundary after block 10.
- Sleep: 30 replay updates, CE + KD (`beta=1`, temperature 2), oracle-weighted task sampling.
- Incoming reset, zero outgoing slices, clear optimizer state.

### Reproducibility and artifacts

Each run writes `config.yaml`, `metrics.parquet`, `events.jsonl`, `checkpoint.pt`, `status.json`, and `environment.txt` under `results/pilots/<pilot>/<run_id>/`. Aggregation writes one CSV and one decision JSON per pilot. `decision.json` must contain the gate inputs, threshold, pass/fail, and run IDs.

## 2. Twelve-hour execution schedule

| Elapsed target | Action | Maximum allocation |
|---:|---|---:|
| 0:00–0:30 | unit tests + 2-minute smoke runs | local/login-safe tests; no training on login node |
| 0:30–1:00 | CPU/GPU timing calibration | 2 short jobs |
| 1:00–3:30 | Pilot A phase diagram | 24 runs, parallel array |
| 2:00–4:00 | Pilot B future-demand value | 8 runs, parallel with A |
| 3:30–5:00 | Pilot C score validity | 4 trained checkpoints + batched ablations |
| 4:00–6:00 | Pilot D compression isolation | 12 runs |
| 5:30–9:30 | Pilot E oracle end-to-end | 12 runs |
| 9:30–11:00 | paired aggregation, plots, decision report | 1 CPU job |
| 11:00–12:00 | one targeted retry only | failed/ambiguous cells |

Do not wait for one stage before submitting independent later stages. Pilot E depends on the width selected by A; prepare manifests in advance and generate its final rows immediately when A completes.

## 3. Pilot A — capacity phase diagram

### Question

Is there an intermediate width that represents the lifetime mixture offline but is poorly used online?

### Exact grid

| Factor | Values |
|---|---|
| Width | `10, 25, 50, 100` |
| Method | `offline_mixture, ER, periodic_full_reset` |
| Seeds | `0,1` paired |
| Updates | 4,500 total |
| Schedule | fixed 30-block skew stream for online; IID skew mixture offline |

Total: **24 runs**. Offline receives exactly the online total update count. Full reset occurs every 5 blocks; it is a diagnostic, not a proposed competitor. Evaluate a fixed 2,000-example mixture set every 300 updates.

### Early stopping

- Stop a run for NaN/divergence or if loss exceeds 20 for three evaluations.
- For offline only, stop after three evaluations with accuracy improvement <0.2 percentage points; keep last/best values.
- Cancel widths smaller than a width whose offline accuracy is already <75% after 60% of its budget only if the smaller width is also <75% at that time.

### Gate

For each width compute paired mean final mixture accuracy and gap `offline − ER`.

- **PASS:** at least one width has offline ≥88%, gap ≥8 percentage points, and ER ≥55%.
- **MARGINAL:** offline ≥85% and gap ≥5 points. Continue with that width but mark the phenomenon weak.
- **FAIL-infeasible:** all offline values <85%. Try width 200 once; if offline remains <85%, fix task/model/training design.
- **FAIL-overcapacity/no allocation gap:** every width with offline ≥88% has gap <5 points. Increase task interference (label permutations), reduce buffer to 200, or lengthen the stream; run only one diagnostic retry.

Select the **smallest passing width**, breaking ties by largest gap. Never select using ReCAP performance.

Artifacts: `phase_diagram.csv`, `phase_diagram.png`, `selected_width.json`, learning curves, and `A_decision.json`.

## 4. Pilot B — does future demand matter?

This isolates prospective value at the replay level before neuron compression.

### Exact grid

| Factor | Values |
|---|---|
| Replay policy | `uniform_task, oracle_frequency_weighted` |
| Width | `50` (independent of A) |
| Seeds | `0,1,2,3` paired |
| Stream | common 30-block skew stream |
| Budget | 500 examples, 150 updates/block |

Total: **8 runs**. Both policies use reservoir replacement and the same total buffer. Uniform targets 50/task; oracle targets largest-remainder integer quotas from the skew probabilities, with at least 5 examples for every observed task and remaining slots probability-weighted. Replay minibatches follow those same allocations.

### Gate

Primary paired difference: lifetime pre-update accuracy.

- **PASS:** oracle improves mean by ≥2.0 points and is positive in at least 3/4 paired seeds.
- **MARGINAL:** improvement 1–2 points or positive in only 2/4; increase skew once (dominant two tasks total 80%) and rerun two seeds.
- **FAIL:** improvement <1 point after the stronger-skew retry. Prospective allocation is unlikely to be visible in this protocol; redesign the stream before implementing learned prediction.

Artifacts: `replay_policy_pairs.csv`, per-task curves, buffer allocations over time, and `B_decision.json`.

## 5. Pilot C — Taylor distortion validity

### Exact experiment

Train ER checkpoints at the width selected by A (fallback width 50), seeds `0,1`, through 10 forced task introductions plus 5 skew blocks. For tasks `0,1,5`, use 50 held-out probe examples each. In each hidden layer, select 25 evenly spaced unit indices after a seed-specific random permutation (or all units if fewer than 25): at most 100 unit checks per checkpoint.

For each `(unit, task)` record:

\[
s_{j,k}=|g_j\partial L_k/\partial g_j|,
\quad
a_{j,k}=L_k(g_j=0)-L_k(g_j=1).
\]

Use the same probe batch for both. Restore gates after every ablation. Also compare Taylor's ability to identify the lowest-damage 20% against random ranking.

### Gate

- **PASS:** pooled Spearman \(\rho\ge.40\), positive in both seeds, and Taylor bottom-quintile true damage ≤70% of random bottom-quintile damage.
- **MARGINAL:** \(.20\le\rho<.40\). Test `squared-gradient/Fisher` and activation×outgoing-weight on the same saved checkpoint; adopt the best metric if \(\rho\ge.40\).
- **FAIL:** all cheap metrics have \(\rho<.20\), or ranking selects more harmful units than random. Use empirical ablation for the tiny oracle pilot and do not build the main method around Taylor until redesigned.

Artifacts: row-level `score_vs_ablation.parquet`, scatter/rank plot, per-layer/task correlations, and `C_decision.json`.

## 6. Pilot D — compression in isolation

### Setup and grid

Start from the same seed-matched ER checkpoints used in C. Preserve tasks according to the known skew weights. Each condition masks/recycles 20% per hidden layer.

| Condition | Selection | Sleep |
|---|---|---|
| `importance_prune` | lowest chosen-metric utility | none |
| `importance_consolidate` | same units | 30 CE+KD steps |
| `random_consolidate` | random units | 30 CE+KD steps |
| `importance_ce_only` | same units | 30 CE steps |

Seeds `0,1,2`: **12 runs**. Measure weighted probe NLL/accuracy at four exact moments: before mask, after mask, after sleep, after function-neutral reset.

Define recovery ratio in NLL:

\[
R=\frac{L_{mask}-L_{sleep}}{\max(L_{mask}-L_{before},10^{-8})}.
\]

### Gate

- **PASS:** median \(R\ge.60\%\), final weighted accuracy is within 3 points of pre-mask, and importance+consolidation beats random+consolidation by ≥1 point.
- **MARGINAL:** \(R=30\)–60% or final drop 3–6 points. Retry once with 100 sleep steps and recycle 10%.
- **FAIL:** recovery <30%, final drop >6 points after retry, or reset itself changes logits by >`1e-5`. Fix consolidation/reset before end-to-end work.

Artifacts: stagewise paired plot, recovery table, logit-invariance test output, and `D_decision.json`.

## 7. Pilot E — oracle end-to-end kill test

### Exact grid

Use A's selected width and the common 30-block skew schedule.

| Method | Details |
|---|---|
| `ER` | buffer 500, normal online updates |
| `ER_compute_matched` | same additional 30 replay steps at recycle boundaries, no mask/reset |
| `random_recycling` | same cadence, fraction, and consolidation as ReCAP; random units |
| `oracle_ReCAP` | true generator probabilities, selected importance, consolidation, safe reset |

Seeds `0,1,2`: **12 runs**. If D fails decisively, do not spend the remaining budget on E; issue a NO-GO on the current compression mechanism. If C fails but D can run with empirical ablation, label oracle results “metric-independent upper-bound diagnostic.”

### Primary gate

Compare lifetime pre-update accuracy with paired differences.

- **GO:** oracle ReCAP beats ER by ≥3 points, random recycling by ≥2 points, and compute-matched ER by ≥2 points on the mean; differences are positive in all 3 seeds; no >10-point collapse on dominant tasks immediately after recycling.
- **CONDITIONAL GO:** oracle wins by 1–3 points consistently and D passed. Proceed only with learned predictor + one mechanism refinement; reduce the main grid.
- **NO-GO:** oracle improvement over both ER and random is <1 point, loses on ≥2 seeds, or gain disappears against compute-matched ER. Do not implement learned \(P_t\); identify whether scoring, consolidation, or benchmark incentive is the bottleneck.

Secondary diagnostics: online NLL, dominant/rare task curves, reacquisition area, mask damage/recovery, unit age, and capacity-selection heatmap.

Artifacts: `oracle_pairs.csv`, prefix-average plot, per-task heatmap, mechanism events table, and `E_decision.json`.

## 8. Overall decision logic

```text
A fails → fix representational/interference regime; stop.
A passes, B fails → benchmark does not reward prospective demand; redesign stream; stop.
C fails → replace importance metric; do not claim Taylor mechanism.
D fails → compression-before-recycling mechanism fails; stop.
E fails → NO-GO for ReCAP even with perfect prediction; do not build predictor.
E conditional/pass → implement learned decayed transitions and launch reduced main study.
```

Issue **FULL GO** only if A, B, D, and E pass and C passes with Taylor or a declared replacement. Issue **CONDITIONAL GO** for one marginal gate with a clear repair. Two or more marginal gates count as NO-GO for a two-week workshop timeline.

## 9. Narval execution details

Create separate manifests for A–E and submit arrays with concurrency caps. Example CPU-first pilot script:

```bash
#!/bin/bash
#SBATCH --account=def-ACCOUNT
#SBATCH --time=02:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --array=0-23%12
#SBATCH --output=logs/%A_%a.out
set -euo pipefail
module purge
module load python/3.11
source "$PROJECT/recap/venv/bin/activate"
python -m src.pilots --manifest "$PROJECT/recap/manifests/pilot_A.csv" --row "$SLURM_ARRAY_TASK_ID"
```

Treat module versions/resources as templates: verify them on Narval. Use `$SCRATCH` for dataset cache and checkpoints, `$PROJECT` for manifests/code/final metrics. A controller should read `selected_width.json`, render Pilot E's manifest, and submit it with a dependency on A's aggregation job. Do not train on login nodes. Use completion markers and resubmit only missing rows.

Recommended resource caps:

- MNIST training: 4 CPUs, 8–12 GB RAM, no GPU unless timing calibration gives ≥2× throughput.
- GPU alternative: 1 GPU, 4 CPUs, 12 GB, pack multiple sequential MNIST runs per allocation if permitted and simpler than tiny independent GPU jobs.
- Aggregation: 2 CPUs, 8 GB, 30 minutes.
- Per-run timeout: 2 hours; whole array critical path target: <4 hours.

Current operational details should be checked against official Alliance pages for [Narval](https://docs.alliancecan.ca/wiki/Narval), [running jobs](https://docs.alliancecan.ca/wiki/Running_jobs), [Python](https://docs.alliancecan.ca/wiki/Python), and [storage](https://docs.alliancecan.ca/wiki/Storage_and_file_management).

## 10. Final pilot report

Generate `PILOT_DECISION_REPORT.md` automatically with:

- environment and total CPU/GPU-hours;
- one table containing A–E thresholds, observed paired values, and decisions;
- five required plots;
- failed/cancelled run IDs and retry rationale;
- selected width and selected importance metric;
- exact recommended next action: `FULL_GO`, `CONDITIONAL_GO`, or `NO_GO`;
- if GO, the frozen main defaults; if NO-GO, the single most diagnostic next experiment.

Do not relax thresholds after seeing results. If a software bug invalidates a run, document it and rerun the identical config. Queue delays do not count against scientific runtime, but the submitted workload must remain sized to finish within roughly 12 wall-clock hours once scheduled.

