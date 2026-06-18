# State Aggregation for Decision-Focused Learning

Code for three MDP experiments: inventory management (`mdp_lp`), cliff-walking (`mdp_cw`), and CartPole (`mdp_continuous`). All experiments compare Decision-Focused Learning (DFL) variants against a Two-Stage baseline.

## Installation

**Requirements:** Python 3.10+, PyTorch, Gurobi (academic license), qpth, scipy.

```bash
pip install -r requirements.txt
```

Gurobi academic licenses are free: [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/).

---

## Experiment 1: Inventory Management (`mdp_lp`)

Inventory MDP with demand elasticity. The model predicts demand λ from price features; the LP plans orders.

**Algorithms:** `twostage`, `dfl`, `dfl_qpth`, `dfl_sketch`, `dfl_berthet`

**Smoke test** (< 30 s):
```bash
python -m mdp_lp.run_elasticity_experiments \
    --experiment_type a --cost_case high_margin --algo twostage \
    --n_epochs 2 --n_samples 10 --pretrain_epochs 1
```

**Single run** (paper settings):
```bash
python -m mdp_lp.run_elasticity_experiments \
    --experiment_type b --cost_case high_margin --algo dfl \
    --n_epochs 100 --n_samples 200
```

**Batch run** (all 16 experiments in parallel):
```bash
python -m mdp_lp.run_all_elasticity
```

**Output:** `mdp_lp/results_elasticity/exp_{a,b,c}/{cost_case}/{algo}/epochs_{n}/results.json`

**Key metrics:** `mean_regret`, `lambda_mse`, `test_obj_values`

See [mdp_lp/README.md](mdp_lp/README.md) for full parameter reference.

---

## Experiment 2: Cliff Walking (`mdp_cw`)

4×12 grid-world MDP with weather-regime-dependent slip dynamics. The model predicts slip distributions from weather features; the LP computes the optimal policy.

**Algorithms:** `twostage`, `dfl`, `dfl_qpth`, `dfl_ssa`, `dfl_sketch`

**Smoke test** (< 30 s):
```bash
python -m mdp_cw.run_cliff_experiments \
    --algo twostage --n_samples 20 --n_epochs 2 --pretrain_epochs 5
```

**Single run** (paper settings — v4 hard regime):
```bash
python -m mdp_cw.run_cliff_experiments \
    --algo dfl --regime_set v4 --cliff_penalty -10 \
    --noise_std 2.5 --center_scale 0.88 \
    --ambiguous_pairs 1:2,2:3 \
    --n_samples 120 --n_epochs 50 --pretrain_epochs 60 \
    --loss_weight 500 --aug_lag_rho 1.0 --lr 3e-3 \
    --save_dir mdp_cw/results/dfl_run0
```

**Batch run** (10 seeds, DFL + Two-Stage, concurrency 2):
```bash
python -m mdp_cw.run_v4_hard_experiments \
    --algos ts,dfl --n_seeds 10 --skew_k 7 --n_samples 120 \
    --concurrency 2 --scale_loss_weight_with_skew
```

**Output:** `{save_dir}/results.json` (single run) or `mdp_cw/results_n120/n{N}/k{K}/{algo}_s{seed}/results.json` (batch)

**Key metrics:** `regret_mean` (pred mean − oracle mean on test set), `slip_ce`

See [mdp_cw/README.md](mdp_cw/README.md) for full parameter reference.

---

## Experiment 3: CartPole (`mdp_continuous`)

Continuous-state CartPole with offline DFL. Features are learned via behavior cloning; the LP optimizes occupancy over a discretized state space.

**Algorithms:** `neural_dfl` (DFL-LP), `dfl_qpth` (DFL-QP), `two_stage_mbrl` (MBRL baseline)

**Smoke test** (< 30 s):
```bash
python -m mdp_continuous.make_figure1_cartpole \
    --seeds 1 --n_iterations 2 --n_workers 1 \
    --pretrain_epochs_dfl 5 --pretrain_epochs_ts 5 \
    --save_path /tmp/cartpole_test.png
```

**Full experiment** (paper settings, 8 seeds in parallel):
```bash
python -m mdp_continuous.make_figure1_cartpole \
    --seeds 1 2 3 4 5 6 7 8 --n_iterations 16 \
    --n_workers 8 --save_path mdp_continuous/figure1_cartpole.pdf
```

**Multi-solver run** (lp / gurobi_qp / qpth / two_stage):
```bash
python -m mdp_continuous.run_dfl_vs_qpth \
    --seeds 1 2 3 4 5 --solvers lp qpth two_stage
```

**Output:** `mdp_continuous/results/dfl_vs_qpth/*.json` (cached per seed), `figure1_cartpole.pdf`

**Key metric:** `avg_reward` (episode return on CartPole, max 200)

See [mdp_continuous/README.md](mdp_continuous/README.md) for full parameter reference.

---

## Common Code Infrastructure

| Module | Description |
|--------|-------------|
| `mdp_lp/mdp.py` | MDP → LP formulation (H, r, gamma) |
| `mdp_lp/solvers/differentiable_lp.py` | Differentiable LP layer (basis-sensitivity backward) |
| `mdp_lp/solvers/dense_lp_solver.py` | Gurobi-backed LP forward |
| `mdp_lp/soft_state_aggregation.py` | Soft state aggregation (`SoftAggDense`) |
