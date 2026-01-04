# mdp_lp: MDP Linear Programming Solver

PyTorch implementation for solving discounted Markov Decision Processes via Linear Programming, with support for differentiable optimization and decision-focused learning.

## Overview

This library implements:
1. MDP to LP formulation: Converts transition probabilities and rewards into linear programming constraints
2. LP solvers: Dense (Gurobi), differentiable (basis-restricted sensitivity analysis)
3. Training methods: Direct Feedback Learning (DFL) and Two-Stage training for learning MDP parameters

## Installation

**Prerequisites:**
- Python 3.7+
- PyTorch
- Gurobi Optimizer (academic license available)
- scipy

**Install:**
```bash
pip install torch scipy gurobipy
```

Gurobi academic licenses are free. See [Gurobi Academic Program](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Running Experiments

### Elasticity-Based Inventory Experiments

**Run all experiments (parallel execution):**
```bash
python -m mdp_lp.run_all_elasticity
```

**Run single experiment:**
```bash
python -m mdp_lp.run_elasticity_experiments \
    --experiment_type a \
    --cost_case high_margin \
    --algo dfl \
    --n_epochs 100 \
    --n_samples 100
```

**Generate comparison plots:**
```bash
python -m mdp_lp.create_comparison_plots
```

### Experiment Configuration

**16 total experiments:**
- Experiment (a): β₂=0, p constant → 4 experiments (2 cost cases × 2 algorithms)
- Experiment (b): β₂=0, p lognormal → 4 experiments (2 cost cases × 2 algorithms)
- Experiment (c): β₂≠0, p and p_related lognormal → 8 experiments (2 cost cases × 2 algorithms × 2 β₂ values)

**Elasticity model:** `ln(λ) = β₀ + β₁ ln(p) + β₂ ln(p_related) + ε`

**Output files per experiment:**
- `results.json`: Complete metrics and data
- `train_reward_per_epoch.png`: Training cost over epochs
- `lambda_histograms.png`: Predicted vs true lambda distributions
- `test_obj_boxplot.png`: Test set optimal values (single method)
- `comparison_boxplot.png`: DFL vs Two-stage comparison (parent directory)

## Results Interpretation

### Plot Descriptions

**train_reward_per_epoch.png**: Training cost over epochs (negative values represent costs). Decreasing cost indicates improving performance.

**lambda_histograms.png**: Predicted vs true lambda distributions. Overlapping histograms indicate accurate predictions.

**test_obj_boxplot.png**: Distribution of optimal values on test set. Higher values (less negative) indicate better performance.

**comparison_boxplot.png**: Side-by-side comparison of DFL vs Two-stage on test set. Primary metric for method comparison.

### Key Metrics

- **Lambda prediction error**: `|λ_pred - λ_true|`, should decrease over epochs
- **obj_loss**: `|obj_hat - obj_true|`, measures prediction quality
- **Test optimal values**: Compare mean/std between DFL and Two-stage

## Performance

**Bottlenecks:**
- LP solving (Gurobi): ~70% of time, CPU-only
- Neural network training: ~20% of time
- Data preparation: ~10% of time

**Recommended:** Use parallel execution via `run_all_elasticity.py` which automatically distributes experiments across CPU cores.

**Expected runtime:**
- Per experiment (100 samples, 100 epochs): 1-2 hours
- All 16 experiments in parallel (10 workers): 2-3 hours total

**GPU:** Not used. Sparse tensors require CPU, and LP solving is CPU-only.

## Parameters

**Elasticity model:**
- `beta_0 = 2.0`, `beta_1 = -1.5`, `beta_2 = 0.0` (varies by experiment), `sigma_epsilon = 0.1`
- Price: `p_constant = 10.0` (exp a), or lognormal(mean=10.0, std=0.5) (exp b, c)
- Related price: lognormal(mean=8.0, std=0.5) (exp c only)

**MDP structure:**
- States: 151 (b_max=50, s_max=100, inventory range -50 to +100)
- Actions: 21 (q_max=20, order quantities 0 to 20)
- Demand: Poisson with max d_max=30

**Cost parameters:**
- High margin: `foc=1.0, voc=0.5, hc=0.1, bc=0.2` (holding cost << revenue)
- Low margin: `foc=1.0, voc=0.5, hc=20.0, bc=0.2` (holding cost >> revenue)

**Training:**
- Epochs: 100, Samples: 100, Train/test split: 80/20
- Learning rate: 0.001, Hidden dimension: 32

## Notes

**Negative values in results:** Represent costs, not rewards. More negative indicates higher costs. High margin experiments show ~-300 to -400, low margin ~-37,000.

**Training cost trends:** Cost may decrease as the model learns the true cost structure. Monitor lambda prediction error and obj_loss to confirm learning.

**Elasticity model:** Ground truth lambda is generated from `ln(λ) = β₀ + β₁ ln(p) + β₂ ln(p_related) + ε`. The neural network does not know this functional form and learns a black-box mapping from prices to lambda.

## Code Structure

**Core modules:**
- `mdp.py`: MDP to LP formulation
- `solvers/`: LP solvers (dense, differentiable)
- `inventory_elasticity.py`: Inventory MDP creation from lambda
- `elasticity_model.py`: Ground truth model and neural network predictor
- `train_elasticity.py`: Training and evaluation logic

**Experiment scripts:**
- `run_all_elasticity.py`: Run all 16 experiments in parallel
- `run_elasticity_experiments.py`: Run single experiment
- `create_comparison_plots.py`: Generate comparison plots

## Results Location

Results are stored in `mdp_lp/results_elasticity/` with structure:
```
exp_X/cost_case/algorithm/epochs_100/
```

Final organized results are copied to `mdp_lp/final_results/` (see `final_results/README.md`).
