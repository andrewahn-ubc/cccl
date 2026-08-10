# ReCAP Workshop Project: Complete Implementation and Paper Specification

## 0. Mission and scope

Build and evaluate **ReCAP (Recurrence-aware Capacity Allocation and Plasticity)**, a continual learner for recurring, nonstationary task streams under a fixed network and replay-memory budget. ReCAP predicts future task demand, estimates which hidden units support each task, compresses future-relevant behavior into a surviving core, and recycles low-future-value units.

The workshop claim is deliberately narrow:

> Under a hard structured-capacity budget, prospective allocation can improve finite-lifetime online performance when task demand is unequal and changes over time.

Do not claim that ReCAP globally optimizes a nonlinear network, that width is Shannon information capacity, or that pruning/resetting itself is novel. Task identities and boundaries are known during training. Supervised learning is the main study; RL is out of scope unless all core experiments are complete.

## 1. Research framing

Ground the paper in these three supplied works:

1. **Continual Learning as Computationally Constrained Reinforcement Learning.** Retention, plasticity, transfer, forgetting, and rapid relearning are means to the lifetime objective, not ends. Obsolete knowledge need not be retained; recurring knowledge may be forgotten if cheap to relearn. Use finite-lifetime pre-update performance as the empirical objective.
2. **Capacity-Constrained Continual Learning.** This provides the clean resource-allocation analogy: under an information constraint, capacity should be assigned according to marginal value. ReCAP transfers this principle to a fixed neural architecture using local, measured deletion distortion—not a formal mutual-information budget.
3. **Continual Learning via Neural Pruning (CLNP).** Pruning, post-pruning fine-tuning, protected pathways, packing into unused capacity, and graceful degradation motivate compression before reuse. CLNP is retrospective/permanent packing; ReCAP adds predicted future demand and allows formerly useful capacity to be reclaimed.

Nearby methods already establish replay, importance-based protection, sparse regrowth, and selective reset. The novelty must therefore be the combination:

> decayed task-dynamics estimation → discounted future occupancy → task-conditioned unit distortion → future-weighted consolidation → selective function-neutral recycling.

The central comparison is **keep everything** (ER/protection), **reset everything** (periodic full reset), and **selectively compress/recycle** (ReCAP). Full reset is a diagnostic extreme: if it beats preservation, accumulated representations are obstructing future learning.

## 2. Hypotheses

- **H1 (allocation-limited regime):** At intermediate width, offline lifetime-mixture training performs well while ordinary continual training performs materially worse.
- **H2 (prospective allocation):** Learned/oracle future demand improves lifetime performance under skewed recurrence and regime shifts, but not materially under uniform recurrence.
- **H3 (compression):** Consolidation before recycling preserves more future-weighted behavior than immediate deletion at the same recycle rate.
- **H4 (adaptation):** Decayed transition estimates let ReCAP abandon historically important but newly obsolete knowledge after a regime shift.
- **H5 (mechanism):** Improvements are not explained solely by replay, extra updates, random fresh units, or oracle access.

## 3. Formal setting and objective

There are tasks \(k\in\{1,\ldots,K\}\), a block-level task identity \(z_b\), model \(f_\theta(x,k)\), fixed parameter architecture, and replay buffer \(\mathcal B\) of fixed total size \(B\). Evaluate predictions before each online update.

Primary objective:

\[
J_T=\frac{1}{T}\sum_{t=1}^T \mathbf 1[\arg\max f_{\theta_t}(x_t,z_t)=y_t].
\]

Also record pre-update negative log-likelihood. The benchmark distribution—not uniform retrospective testing—defines value.

### 3.1 Future demand

Maintain decayed transition counts. At boundary \(i\to j\):

\[
N\leftarrow\lambda_N N,\qquad N_{ij}\leftarrow N_{ij}+1.
\]

With Dirichlet smoothing:

\[
\widehat P_{ij}=\frac{N_{ij}+\alpha}{\sum_l N_{il}+\alpha K}.
\]

From current task \(z_t\), forecast normalized discounted occupancy:

\[
q_t(k)=\frac{1}{Z}\sum_{h=1}^{H}\gamma^{h-1}[e_{z_t}^{\top}\widehat P^h]_k.
\]

Implement `uniform`, `learned`, and `oracle` providers behind one interface. Oracle uses the generator's current true transition process, never the realized future sequence. Default: \(\alpha=0.5\), \(H=10\), \(\gamma=0.9\), \(\lambda_N=0.97\). Ablate these values.

### 3.2 Task-conditioned distortion

Attach a non-trainable scalar gate \(g_j=1\) after every hidden ReLU unit (MLP) or convolutional output channel (CNN). For task \(k\):

\[
d_{j,k}=\mathbb E_{(x,y)\sim\mathcal B_k}\left|g_j\frac{\partial\ell(x,y)}{\partial g_j}\right|.
\]

Average per example, then normalize scores within each layer (divide by layer median plus \(10^{-8}\)) so scale does not force recycling from one layer. Validate with empirical deletion:

\[
d^{\mathrm{abl}}_{j,k}=\mathcal L_k(g_j=0)-\mathcal L_k(g_j=1).
\]

Use absolute Taylor scores for selection; log signed scores too. Candidate alternative metrics: activation×outgoing-weight utility, Fisher gate score, empirical ablation, and random.

### 3.3 Future utility and selection

\[
U_j(t)=\sum_k q_t(k)d_{j,k}.
\]

Optionally test relearning burden only after the base result works:

\[
U_j^+(t)=\sum_k q_t(k)\tau_k d_{j,k},
\]

where \(\tau_k\) is the normalized area between early-return loss and the task's settled loss. Select the lowest \(\rho\) fraction **within each recyclable layer**, with minimum one unit only when the layer has at least \(1/\rho\) units. Default \(\rho=0.20\).

### 3.4 Compress, then recycle

At selected boundaries:

1. Freeze a teacher snapshot of the pre-mask model.
2. Compute \(q_t\), \(d_{j,k}\), and \(U_j\).
3. Mask the selected low-utility units.
4. Optimize surviving parameters for \(S\) sleep updates on stratified replay, leaving masked parameters frozen.
5. Use

\[
\mathcal L_{sleep}=\sum_k q_t(k)\,\mathbb E_{\mathcal B_k}
[\ell_{CE}(y,f_{core}(x))+\beta T_d^2D_{KL}(p_{teacher}^{T_d}\Vert p_{core}^{T_d})].
\]

6. Reinitialize selected units' incoming weights with the layer's original initializer; zero biases.
7. Set their outgoing weights to zero, including the corresponding input slices in the next layer. This makes their immediate contribution zero.
8. Clear optimizer momentum/Adam state for every reset parameter slice.
9. Unmask and resume online training.

Defaults: \(S=30\), \(\beta=1\), distillation temperature \(T_d=2\), recycle every task boundary after a two-boundary warm-up. Do not update batch-normalization statistics during scoring; prefer architectures without batch norm for the core study.

## 4. Local theory statement

Under an additive first-order deletion approximation,

\[
\Delta\mathcal L(S)\approx\sum_{j\in S}U_j.
\]

For equal-cost units and a fixed retained count, keeping the highest-\(U_j\) units minimizes predicted future-weighted local distortion. For unequal costs \(c_j\), selection becomes a knapsack problem. State this as a local surrogate guarantee, not global neural optimality.

Connect this to a future-weighted diagonal allocation model:

\[
\min_{B_i\ge0}\sum_iq_i e^{-2B_i}(\Sigma_i-M_i),\quad \sum_iB_i=B,
\]

whose KKT solution is

\[
B_i^*=\frac12\left[\log\frac{2q_i(\Sigma_i-M_i)}{\eta}\right]_+.
\]

This exact toy result establishes the principle: future frequency multiplies useful predictable signal. The neural score substitutes empirical loss distortion for known analytic distortion.

## 5. Software architecture and correctness requirements

Use Python, PyTorch, Hydra/OmegaConf or plain YAML, pandas, scipy, seaborn, and pytest. Suggested layout:

```text
recap/
  configs/{benchmark,method,schedule,cluster}/
  src/{data,schedules,models,replay,methods,metrics,analysis}/
  scripts/{train,pilot,aggregate,plot,make_manifest}.py
  slurm/{cpu_array,gpu_array,aggregate}.sbatch
  tests/
  results/raw/  results/derived/  figures/  paper/
```

Every run must be reconstructible from one resolved config. Save: git commit, dirty flag, hostname, package versions, seed, task definitions, full schedule, transition matrices, run timestamps, SLURM IDs, and hardware. Use deterministic data-loader seeding. Pair seeds across methods: the same seed implies identical tasks, stream, initialization where compatible, and buffer insertion randomness.

Mandatory tests:

- transition rows sum to one; decay and oracle switching are correct;
- future occupancy matches hand-computed two/three-state chains;
- masking one known unit changes only the expected activations;
- Taylor scoring produces one score per recyclable unit without modifying weights;
- zeroed outgoing slices preserve logits after recycling to tolerance \(10^{-6}\);
- optimizer state is cleared only for reset slices;
- replay capacity never exceeds \(B\) and per-task sampling is correct;
- pre-update metrics are recorded before optimization;
- resume-from-checkpoint reproduces uninterrupted metrics.

## 6. Benchmarks and streams

### 6.1 Weighted LQG/synthetic allocation

Thirty independent scalar subsystems with heterogeneous dynamics, noise, observability, and \(q_i\). Compare uniform bits, variance-proportional allocation, learned-\(q\) water filling, and oracle-\(q\). Plot objective vs total budget, allocations, learned–oracle gap, and adaptation to a mid-run shift.

### 6.2 Recurring Permuted MNIST (main)

- 10 tasks; fixed random input permutation and label permutation per task.
- Task one-hot appended to input; shared 10-class head.
- Two-hidden-layer ReLU MLP; widths `16,32,64,128,256` for phase diagram, then retain 2–3 informative widths.
- 50 blocks × 500 minibatch updates; batch 64; buffer 500 total.
- Equal per-task reservoir allocation for main comparisons. Future-aware replay allocation is an ablation, not part of base ReCAP.
- Five paired evaluation seeds after hyperparameters are frozen.

### 6.3 Recurring Split CIFAR-100 (confirmation)

- 10 tasks × 10 classes; known-task fixed multi-head output.
- 30 visits × 200 updates (or exactly one epoch; choose once and keep fixed).
- Small four-layer CNN, no batch norm; recycle channels.
- Two widths selected from an offline-vs-ER pilot; buffer 500 and optionally 1000.
- Three paired seeds.
- Run only ER, DER++, ER+CBP, CLNP-style, ReCAP-uniform, ReCAP-learned, and (one width) oracle.

### 6.4 Schedule families

1. **Uniform:** all tasks equally likely.
2. **Skewed:** tasks 0–2 receive 60%, 3–5 receive 25%, 6–9 receive 15%, divided equally within groups.
3. **Anchor+distractors:** `A,B1,A,B2,...`; A recurs, distractors rarely or never.
4. **Regime shift:** first half favors 0–2; second half favors 3–5. Use a configurable abrupt shift; a gradual interpolation is a stretch ablation.

Persist schedule-generation parameters and realized sequences. Development and evaluation schedules must be disjoint.

## 7. Baselines and fairness

Essential MNIST methods:

- offline lifetime-mixture reference (diagnostic, not an online competitor);
- online fine-tuning;
- ER with identical total replay budget and online updates;
- periodic full reset at the same boundaries used by ReCAP;
- random recycling with identical recycle rate/timing/sleep compute;
- Continual Backprop with matched replacement rate;
- CLNP or explicitly labelled CLNP-style implementation;
- online EWC;
- ReCAP-uniform, learned, and oracle.

For each comparison report both natural compute and a **compute-matched** version. Random recycling must receive the same sleep batches; a no-op/ER-extra-update control receives the same number of gradient steps without masking. No baseline may use realized future tasks. Offline mixture gets the same architecture and total update count, sampling tasks according to true lifetime frequencies.

## 8. Ablations

Prioritized:

1. learned vs uniform vs oracle \(q\);
2. consolidation on/off;
3. CE only vs KD only vs CE+KD;
4. future-weighted vs uniform sleep sampling/loss;
5. Taylor vs random vs empirical deletion (MNIST subset);
6. transition decay `1.0, .99, .97, .90` and sliding-window alternative;
7. recycle fraction `.05,.10,.20,.30` and cadence every `1,2,5` boundaries;
8. sleep updates `0,10,30,100`;
9. buffer `100,500,1000` and equal vs future-aware allocation;
10. horizon `1,5,10,20` and discount `.5,.9,1.0`;
11. zero outgoing vs ordinary random outgoing initialization;
12. optional relearning burden \(\tau_k\).

Run broad ablations on one allocation-limited MNIST width and two schedules; only validate the top mechanisms on CIFAR.

## 9. Metrics, statistics, and diagnostics

Primary: lifetime pre-update accuracy and NLL. Secondary:

- prefix-average accuracy/NLL with regime boundary marked;
- per-task return accuracy and reacquisition area/time-to-threshold;
- forward transfer/new-task learning speed;
- future-frequency-weighted retained accuracy;
- standard final average accuracy and forgetting (clearly secondary);
- occupancy prediction: TV distance/KL, top-task accuracy, learned–oracle performance gap;
- Spearman and Kendall correlation of Taylor vs empirical deletion, plus top/bottom-quartile precision;
- immediate mask damage, post-sleep recovery ratio, post-reset logit drift;
- utility/selection histograms per task and layer;
- wall time, GPU-hours, updates, peak memory, and replay bytes.

Use paired seeds and report mean, standard error, and 95% paired bootstrap confidence intervals for method differences. Avoid treating minibatches as independent replicates. Provide per-seed points. If formal tests are included, use paired permutation tests and control the small prespecified family of primary comparisons (Holm correction). Report effect sizes regardless of significance.

## 10. Required plots and tables

1. Capacity phase diagram: offline, ER, CBP, CLNP-style, and ReCAP vs width.
2. Lifetime prefix-average curves for skew and regime shift.
3. Post-shift per-task heatmap showing capacity moving from old to new dominant tasks.
4. Learned/oracle/uniform comparison with occupancy-estimation error.
5. Compression mechanism: before mask → after mask → after sleep → after reset.
6. Taylor score vs empirical deletion scatter/rank plot.
7. Recycle-rate × sleep-update heatmap.
8. Compute/performance Pareto plot.
9. Main results table with paired CIs; compact ablation table.

Plots must be generated exclusively from immutable raw logs through a checked-in aggregation script. Never transcribe values manually.

## 11. Narval / Alliance orchestration

Do not hard-code account names, module versions, partitions, or GPU model. Put them in environment variables/config and verify on Narval before submission. Keep source and durable summaries in `$PROJECT`; stage datasets/checkpoints and high-churn logs in `$SCRATCH`; copy final raw metrics and checkpoints needed for reproducibility back to `$PROJECT`. Scratch is not a backup and may be purged.

Create a manifest CSV with one row per run and stable `run_id = hash(resolved_config)`. Submit job arrays in waves. A generic script:

```bash
#!/bin/bash
#SBATCH --account=def-ACCOUNT
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-99%20
#SBATCH --output=logs/%A_%a.out
set -euo pipefail
module purge
module load python/3.11
source "$PROJECT/recap/venv/bin/activate"
python -m src.train --manifest "$PROJECT/recap/manifests/main.csv" --row "$SLURM_ARRAY_TASK_ID"
```

Use CPU arrays for MNIST/LQG if profiling shows GPUs are slower after queue time; GPU arrays for CIFAR. Start with a 10-minute smoke job, then one full run, then arrays. Implement atomic checkpoints, `SIGUSR1` pre-timeout handling, resume, per-run completion markers, and a resubmitter that selects missing/failed run IDs rather than rerunning the array. Never download datasets independently from every worker; stage once. Limit simultaneous array jobs to protect the filesystem.

Suggested waves:

1. tests + 1 smoke seed;
2. pilot phase diagram;
3. main MNIST methods on fixed hyperparameters;
4. MNIST ablations;
5. selected CIFAR confirmation;
6. aggregation/plots on CPU.

Before submission, consult current official Alliance documentation for [job scheduling](https://docs.alliancecan.ca/wiki/Running_jobs), [Narval](https://docs.alliancecan.ca/wiki/Narval), [Python environments](https://docs.alliancecan.ca/wiki/Python), and [storage](https://docs.alliancecan.ca/wiki/Storage_and_file_management). Record the exact working module choices in the repository.

## 12. Decision rules for the full study

Proceed only after the separate pilot specification passes. During the main study, downgrade claims if learned ReCAP does not approach oracle, if gains vanish under compute matching, or if gains occur only on one chosen schedule/width. A useful negative result is still publishable if the capacity phase diagram and oracle analysis clearly identify when prospective allocation can and cannot help.

## 13. Workshop paper drafting instructions

Target an 8–9 page main paper plus appendix unless the venue says otherwise. Draft only after `results/derived/final_metrics.csv` and figure manifests are frozen.

Suggested structure:

1. **Abstract:** problem, ReCAP mechanism, benchmarks, one quantified primary result, limitation.
2. **Introduction:** lifetime objective; allocation-limited regime; three contributions (benchmark/diagnostic, method, evidence).
3. **Related work:** computationally constrained CL, pruning/packing, replay/protection, plasticity/resetting, predictive resource allocation. State overlaps plainly.
4. **Setting:** known task boundaries/IDs, fixed network/buffer, pre-update lifetime objective.
5. **Method:** demand prediction, distortion, local allocation argument, sleep compression, safe recycling.
6. **Experiments:** paired design, streams, fairness, compute, development/evaluation split.
7. **Results:** capacity diagram first, main oracle/learned result second, mechanism/ablations third, CIFAR confirmation last.
8. **Limitations:** task IDs, finite library, replay dependence, local/additive score, synthetic recurrence, no global optimality, limited seeds/compute.
9. **Conclusion:** prospective capacity management, not universal forgetting prevention.

Appendix: derivation, algorithms, complete configs, extra curves, failed settings, resource use, seeds, environment lock, and reproducibility checklist. Every numerical claim must be traceable to a table cell and run IDs. Use “supports/is consistent with,” not causal language unsupported by interventions. Do not cherry-pick widths after looking at test seeds: select using development streams and freeze.

## 14. Definition of done

- Reproducible repository, tests, environment lock, data/setup instructions.
- Raw immutable logs and derived tidy tables.
- Capacity phase diagram demonstrating or rejecting an allocation-limited regime.
- Fair main comparisons with paired seeds and CIs.
- Mechanism and compute-matching ablations.
- CIFAR confirmation if pilots and MNIST succeed.
- All required figures/tables and a paper draft with honest limitations.
- `README_REPRODUCE.md` containing exact commands from environment creation through paper figures.

