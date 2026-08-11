# ReCAP 12-hour pilot decision report

**Recommendation: `NO_GO`**

Selected width: **50**. Selected importance metric: **empirical**.

## Gate summary

| Pilot | Threshold | Observed gate inputs | Decision |
|---|---|---|---|
| A | `{'marginal': {'gap': 0.05, 'offline': 0.85}, 'pass': {'er': 0.55, 'offline': 0.88, 'offline_minus_er': 0.08}}` | `{'10': {'ER': 0.76125, 'gap': 0.08025000000000004, 'offline_mixture': 0.8415, 'periodic_full_reset': 0.6577500000000001, 'seed': 0.5}, '100': {'ER': 0.91925, 'gap': 0.015000000000000013, 'offline_mixture': 0.93425, 'periodic_full_reset': 0.7735000000000001, 'seed': 0.5}, '25': {'ER': 0.86075, 'gap': 0.02749999999999997, 'offline_mixture': 0.88825, 'periodic_full_reset': 0.7244999999999999, 'seed': 0.5}, '50': {'ER': 0.888, 'gap': 0.02650000000000008, 'offline_mixture': 0.9145000000000001, 'periodic_full_reset': 0.7617499999999999, 'seed': 0.5}}` | **FAIL_OVERCAPACITY** |
| B | `{'marginal_mean_difference': 0.01, 'pass_mean_difference': 0.02, 'pass_positive_pairs': 3}` | `{'base_mean_paired_difference': 0.002389756944444432, 'base_pairs': {'0': 0.001128472222222232, '1': 0.004177083333333331, '2': 0.004295138888888883, '3': -4.166666666671759e-05}, 'base_positive_pairs': 3, 'strong_skew_error': 'triggered retry produced no complete rows'}` | **FAIL** |
| C | `{'bottom_damage_ratio': 0.7, 'positive_both_seeds': True, 'spearman': 0.4}` | `{'reason': 'Pilot C was skipped because an upstream scientific gate failed'}` | **FAIL** |
| D | `{'importance_minus_random': 0.01, 'max_accuracy_drop': 0.03, 'recovery_ratio': 0.6}` | `{'reason': 'Pilot D was skipped because an upstream scientific gate failed'}` | **FAIL** |
| E | `{'oracle_minus_compute_matched': 0.02, 'oracle_minus_er': 0.03, 'oracle_minus_random': 0.02}` | `{'reason': 'Pilot E was skipped because an upstream scientific gate failed'}` | **NO_GO** |

## Environment and cost

Measured allocation use: approximately **5.02 CPU-hours** and **0.00 GPU-hours**. Exact per-run package, host, Slurm, and hardware details are stored in each `environment.txt`.

## Required figures

- [Pilot A phase diagram](A/phase_diagram.png)
- [Pilot B per-task curves](B/per_task_curves.png)
- [Pilot C score/ablation plot](C/score_vs_ablation.png)
- [Pilot D stagewise compression](D/stagewise_compression.png)
- [Pilot E lifetime prefix average](E/prefix_average.png)
- [Pilot E per-task heatmap](E/per_task_heatmap.png)

## Failed, cancelled, and retried runs

`00bf12eedf32efab`, `06e976691c321df6`, `0b0e9cc625af22a2`, `19d072ba9de50913`, `247203111da874c1`, `2be1d08503583ad2`, `3070dbd6fca44a88`, `3dc63ad2f963f8d9`, `481a284e018c0b15`, `5054f1424e9c83ec`, `50e0c0ad9166cf47`, `72340a03c7a9acd9`, `80e79b560a5fd58c`, `87fd8baa013e5dfd`, `8832a0f20b1522cd`, `8ea388c4f75ae8d0`, `914792891e93de5d`, `9ec9e1fbdbba9b0f`, `ac5b6b2ba10fe6e2`, `b5195048fe62adf6`, `b6c801ede52a0fd6`, `b9c6c8b67d2e8daa`, `c6e4f020e4a8e8ad`, `c92a74f991469019`, `cea51073eab58419`, `dae753a83f24ee4e`, `e820c09b3ce26360`, `ed4565bafdf0eb29`, `f5a05f0aa748d61a`, `f79c436b8ff9d391`

Targeted retries are permitted only for the predeclared marginal cases. A software-invalid run must be rerun with its identical manifest row; the resubmission helper records that rationale.

## Exact next action

Run one width-200 offline feasibility diagnostic before changing the task/model regime.

If proceeding, the frozen defaults are: buffer 500; batch 64; Adam 1e-3; oracle occupancy H=10, gamma=.9; 20% layer-wise recycling after block 10; 30 CE+KD sleep updates; beta=1; temperature=2; function-neutral reset.
