# mdp_cw: Cliff-Walking MDP

4×12 grid-world MDP with weather-regime-dependent slip dynamics. The model predicts per-state slip distributions from weather features; the LP computes the optimal occupancy policy.

## Overview

- **States:** 48 (4 rows × 12 cols); START = (3,0), GOAL = (3,11), CLIFF = row 3 cols 1–10
- **Actions:** 4 (UP/DOWN/LEFT/RIGHT)
- **Discount:** γ = 0.95
- **Regimes:** 4–8 weather regimes with distinct slip distributions
- **Features:** 8-dim weather vectors drawn from regime-specific Gaussian clusters

## Algorithms

| Slug | Description |
|------|-------------|
| `twostage` | Supervised CE on slip distributions; no LP in training |
| `dfl` | Differentiable LP via basis-sensitivity backward |
| `dfl_qpth` | QP-regularized surrogate via qpth |
| `dfl_ssa` | DFL on a learned soft state aggregation (S=48 → M clusters) |
| `dfl_sketch` | Sketched LP: average of K row-reduced LPs |

## Running Experiments

### Quick Smoke Test

Runs in under 30 seconds. Verifies imports and training loop.

```bash
python -m mdp_cw.run_cliff_experiments \
    --algo twostage \
    --n_samples 20 --n_epochs 2 --pretrain_epochs 5
```

### Single Run (paper settings — v4 hard regime)

```bash
python -m mdp_cw.run_cliff_experiments \
    --algo dfl \
    --regime_set v4 --cliff_penalty -10 \
    --noise_std 2.5 --center_scale 0.88 \
    --ambiguous_pairs 1:2,2:3 \
    --n_samples 120 --n_epochs 50 --pretrain_epochs 60 \
    --loss_weight 500 --aug_lag_rho 1.0 --lr 3e-3 \
    --save_dir mdp_cw/results/dfl_run0
```

### Batch Run (multiple seeds, paper scale)

Runs `ts` and `dfl` for 10 seeds with concurrency 2. Skips runs with an existing `results.json`.

```bash
python -m mdp_cw.run_v4_hard_experiments \
    --algos ts,dfl \
    --n_seeds 10 --skew_k 7 --n_samples 120 \
    --concurrency 2 --scale_loss_weight_with_skew
```

To also run `qpth` and `ssa`:

```bash
python -m mdp_cw.run_v4_hard_experiments \
    --algos qpth,ssa \
    --n_seeds 10 --skew_k 7 --n_samples 120 \
    --concurrency 2 --scale_loss_weight_with_skew \
    --ssa_n_agg 24 --ssa_fixed_w --ssa_spatial_init
```

### Data Scaling Ablation

```bash
python -m mdp_cw.run_v4_hard_experiments \
    --n_samples_sweep 20,40,80,120,160 --skew_k 7 \
    --n_seeds 10 --concurrency 2 --scale_loss_weight_with_skew
```

## Output

**Single run** (via `--save_dir`):
```
{save_dir}/
  run_config.json        # all CLI args
  results.json           # metrics
  *.png                  # per-epoch training curves
```

**Batch run** (via `run_v4_hard_experiments`):
```
mdp_cw/results_n120/
  n{N}/k{K}/
    ts_s{seed}/results.json
    dfl_s{seed}/results.json
    qpth_s{seed}/results.json
    ssa_s{seed}/results.json
```

**Key metrics in `results.json`:**
- `regret_mean`: mean(pred policy value − oracle value) on test set (lower is better)
- `oracle_mean`: mean oracle LP value on test set
- `pred_mean`: mean predicted-policy value on test set
- `slip_ce`: cross-entropy between predicted and true slip distributions

## Summarizing Results

After a batch run:

```bash
python -m mdp_cw.summarize_paired_runs \
    --root mdp_cw/results_n120/n120 --k 7 --n_seeds 10
```

## Generating Publication Figures

```bash
python -m mdp_cw.plot_report_three_panel_best_nk \
    --flat_root mdp_cw/results_n120/n120 \
    --k 15 --n_seeds 10 --n_label 200 \
    --out_dir mdp_cw/selected_plots
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--regime_set` | `v1` | Regime library (`v1`–`v6`); `v4` is the paper's hard setting |
| `--n_samples` | 200 | Total samples (half train, half test) |
| `--skew_k` | — | Training skew k:1:1:1 (higher = more imbalanced) |
| `--cliff_penalty` | -100 | Reward for falling off the cliff |
| `--loss_weight` | 1.0 | Weight on CE prediction loss added to DFL loss |
| `--aug_lag_rho` | 0.1 | Augmented Lagrangian penalty for flow constraint |
| `--n_agg` | 12 | Number of soft clusters M for `dfl_ssa` |
| `--pretrain_epochs` | 60 | CE-only warmup epochs before DFL phase |
| `--scale_loss_weight_with_skew` | off | Scale CE loss by skew ratio (recommended at k≥7) |

## MDP Details

- Grid: 4×12 = 48 states; s = row × 12 + col
- START_STATE = 36, GOAL_STATE = 47, CLIFF = {37,...,46}
- Slip encoding: 5 kinds (intended / slip-left / slip-right / back / no-move)
- v4 hard preset: `cliff_penalty=-10`, `noise_std=2.5`, `center_scale=0.88`
