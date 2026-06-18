# mdp_continuous: CartPole MDP

Offline Decision-Focused Learning on continuous-state CartPole. Features are learned via behavior cloning on an offline buffer; the LP plans over a discretized state space via Random Fourier Features.

## Overview

- **Environment:** CartPole-v0 physics (4-dim state, 2 actions)
- **Discount:** γ = 0.99
- **Offline data:** 100–500 trajectories from a suboptimal behavior policy
- **Features:** 128-dim Random Fourier Features (RFF); discretize continuous state → LP state space

## Algorithms

| Name | Description |
|------|-------------|
| `neural_dfl` | DFL-LP: CE pretrain + DFL via sketched LP |
| `dfl_qpth` | DFL-QP: CE pretrain + DFL via Gurobi QP (quadratic regularization) |
| `two_stage_mbrl` | Two-Stage MBRL: fit world model → DQN on frozen model |

## Running Experiments

### Quick Smoke Test

Runs in under 30 seconds. Verifies imports and training loop.

```bash
python -m mdp_continuous.make_figure1_cartpole \
    --seeds 1 --n_iterations 2 --n_workers 1 \
    --pretrain_epochs_dfl 5 --pretrain_epochs_ts 5 \
    --save_path /tmp/cartpole_test.png
```

### Full Experiment (paper settings — 8 seeds)

Runs DFL-LP and Two-Stage MBRL in parallel across seeds. Results are cached under `mdp_continuous/results/dfl_vs_qpth/`.

```bash
python -m mdp_continuous.make_figure1_cartpole \
    --seeds 1 2 3 4 5 6 7 8 \
    --n_iterations 16 --n_workers 8 \
    --save_path mdp_continuous/figure1_cartpole.pdf
```

Re-plot from cached results without re-running:

```bash
python -m mdp_continuous.make_figure1_cartpole \
    --seeds 1 2 3 4 5 6 7 8 \
    --plot_only --save_path mdp_continuous/figure1_cartpole.pdf
```

### Multi-Solver Comparison

Runs multiple solver variants (lp / gurobi_qp / qpth / two_stage) for a given seed list:

```bash
python -m mdp_continuous.run_dfl_vs_qpth \
    --seeds 1 2 3 4 5 --solvers lp qpth two_stage
```

## Output

```
mdp_continuous/results/dfl_vs_qpth/
  curve_neural_*.json          # DFL-LP per-seed reward curves
  curve_two_stage_mbrl_*.json  # MBRL per-seed reward curves
  curve_qpth_*.json            # DFL-QP per-seed reward curves (if --include_qpth)

mdp_continuous/figure1_cartpole.pdf   # Publication figure
mdp_continuous/figure1_cartpole.png
```

Each cache JSON contains:
- `seeds`: list of seeds run
- `rewards_per_seed`: list of per-iteration average rewards per seed
- `lengths_per_seed`: list of per-iteration average episode lengths per seed
- `solver_times_per_seed`: LP/QP solver time per seed (seconds)

**Key metric:** `avg_reward` (discounted episode return on CartPole; max ≈ 200)

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seeds` | `1..8` | Random seeds to run (one process per seed) |
| `--n_workers` | 8 | Parallel workers |
| `--n_iterations` | 16 | Outer DFL iterations |
| `--n_trajectories` | 100 | Offline trajectories from behavior policy |
| `--behavior` | `biased` | Behavior policy (`expert`, `noisy`, `biased`, `random`) |
| `--pretrain_epochs_dfl` | 30 | CE pretrain epochs for DFL |
| `--pretrain_epochs_ts` | 100 | CE pretrain epochs for Two-Stage |
| `--n_features` | 128 | Number of RFF basis functions |
| `--gamma` | 0.99 | Discount factor |
| `--qp_eta` | 0.01 | QP regularization η (for `dfl_qpth`) |
| `--force` | off | Re-run even if cached results exist |

## Notes

- Results are cached by configuration string; use `--force` to re-run with the same hyperparameters.
- The `--two_panel` flag shows both per-iteration dynamics and running-max (deployable quality) side by side.
- GPU is not used; LP solving and RFF estimation are CPU-only.
