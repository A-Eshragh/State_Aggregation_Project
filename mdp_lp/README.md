# mdp_lp: MDP Linear Programming Solver

A PyTorch-based library for solving discounted Markov Decision Processes (MDPs) via Linear Programming (LP), with support for differentiable optimization and multiple training strategies.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Components](#core-components)
  - [MDP Class](#mdp-class)
  - [LP Solvers](#lp-solvers)
  - [Trainers](#trainers)
- [Usage Examples](#usage-examples)
- [Mathematical Background](#mathematical-background)
- [API Reference](#api-reference)

## Overview

`mdp_lp` provides a complete framework for:
1. **Converting MDPs to LP formulations**: Transform transition probabilities and rewards into linear programming constraints
2. **Solving LPs**: Multiple solver backends (dense, sparse, differentiable)
3. **Training predictors**: Two training strategies (DFL and TwoStage) for learning MDP parameters

### Key Features

- ✅ **Batch processing**: Handle multiple MDPs simultaneously
- ✅ **Multiple solvers**: Dense (Gurobi), Sparse (COO format), Differentiable (autograd)
- ✅ **Two training strategies**: Direct Feedback Learning (DFL) and Two-Stage training
- ✅ **Modular design**: Use any PyTorch neural network as predictor
- ✅ **Gradient support**: End-to-end differentiable optimization

## Architecture

### Design Philosophy

`mdp_lp` follows a **modular, separation-of-concerns** design:

- **`mdp_lp` provides**: MDP→LP conversion, LP solvers, training infrastructure
- **You provide**: Neural network architecture (predictor model)

This keeps the library focused on MDP/LP solving while giving you full control over the neural network design.

### Directory Structure

```
mdp_lp/
├── mdp.py              # MDP → LP formulation
├── solvers/
│   ├── dense_lp_solver.py      # Dense LP solver (Gurobi)
│   ├── sparse_lp_solver.py     # Sparse LP solver (COO format)
│   └── differentiable_lp.py    # Differentiable LP (autograd)
├── trainers/
│   ├── dfl_trainer.py          # Direct Feedback Learning trainer
│   └── twostage_trainer.py     # Two-stage training trainer
└── tests/
    └── tests.py                # Comprehensive test suite
```

## Installation

### Prerequisites

- Python 3.7+
- PyTorch
- Gurobi Optimizer (with license)
- scipy (for sparse matrices)

### Install Dependencies

```bash
pip install torch scipy gurobipy
```

**Note**: Gurobi requires a license. Academic licenses are free. See [Gurobi License](https://www.gurobi.com/academia/academic-program-and-licenses/) for details.

## Quick Start

### Basic Usage: Solve an MDP

```python
import torch
from mdp_lp.mdp import MDP
from mdp_lp.solvers import solve_lp_dense

# Define a simple MDP: 2 states, 2 actions
beta = 0.9  # discount factor
mdp = MDP(beta)

# Transition probabilities: P[batch, state, action, next_state]
P = torch.tensor([[
    [[1.0, 0.0], [0.0, 1.0]],  # state 0: a0→s0, a1→s1
    [[0.0, 1.0], [1.0, 0.0]],  # state 1: a0→s1, a1→s0
]])

# Rewards: r[batch, state, action]
r = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])

# Convert to LP formulation
H, r_flat, gamma = mdp.lp_formulation(P, r)

# Solve LP
y_star, basis = solve_lp_dense(H[0], r_flat[0], gamma[0])

# Compute optimal value
v_star = (y_star * r_flat[0]).sum()
print(f"Optimal value: {v_star.item():.4f}")
```

### Training with a Predictor

```python
import torch
import torch.nn as nn
from mdp_lp.trainers import DFLTrainer

# Define your predictor (any PyTorch module)
class MyPredictor(nn.Module):
    def __init__(self, input_dim, S, A):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, S * A * S + S * A)
        )
        self.S, self.A = S, A
    
    def forward(self, x):
        batch = x.size(0)
        out = self.net(x)
        # Split into P and r
        P_flat = out[:, :self.S * self.A * self.S]
        r_flat = out[:, self.S * self.A * self.S:]
        P = torch.softmax(P_flat.view(batch, self.S, self.A, self.S), dim=-1)
        r = r_flat.view(batch, self.S, self.A)
        return P, r

# Create trainer
predictor = MyPredictor(input_dim=10, S=5, A=3)
optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
trainer = DFLTrainer(predictor, beta=0.9, optimizer=optimizer)

# Training step
x = torch.randn(1, 10)  # input features
v_true = torch.tensor([5.0])  # target value (optional)

loss, v_pred = trainer.train_step(x, v_true=v_true)
print(f"Loss: {loss:.4f}, Predicted value: {v_pred.item():.4f}")
```

## Core Components

### MDP Class

The `MDP` class converts MDPs into LP formulations.

#### LP Formulation

For a discounted MDP with:
- States: `S`
- Actions: `A`
- Transition probabilities: `P[s, a, s']`
- Rewards: `r[s, a]`
- Discount factor: `β`

The LP formulation is:

```
maximize:    r^T y
subject to:  H y = γ
             y ≥ 0
```

Where:
- `y ∈ R^{S×A}` is the **occupancy measure** (decision variables)
- `H ∈ R^{S × (S×A)}` is the constraint matrix: `H[s', (s,a)] = δ[s'=s] - β·P[s,a,s']`
- `γ ∈ R^S` is the initial state distribution
- `r ∈ R^{S×A}` is the flattened reward vector

#### API

```python
mdp = MDP(beta, gamma=None)

# Build components
H = mdp.build_H(P)              # (batch, S, S*A)
r_flat = mdp.build_r(r)         # (batch, S*A)
gamma = mdp.build_gamma(...)    # (batch, S)

# Or get all at once
H, r_flat, gamma = mdp.lp_formulation(P, r)
```

**Parameters:**
- `beta` (float): Discount factor (0 < β < 1)
- `gamma` (Tensor, optional): Initial state distribution. If `None`, uses uniform distribution.

**Input shapes:**
- `P`: `(batch, S, A, S)` - transition probabilities
- `r`: `(batch, S, A)` - rewards

**Output shapes:**
- `H`: `(batch, S, S*A)` - constraint matrix
- `r_flat`: `(batch, S*A)` - flattened rewards
- `gamma`: `(batch, S)` - initial distribution

### LP Solvers

Three solver implementations are available:

#### 1. Dense LP Solver

Uses Gurobi with dense matrix representation. Best for small to medium problems.

```python
from mdp_lp.solvers import DenseLPSolver, solve_lp_dense

# Class interface
solver = DenseLPSolver()
y_star, basis = solver.solve(H, r, gamma)

# Function interface
y_star, basis = solve_lp_dense(H, r, gamma)
```

**Returns:**
- `y_star`: `(batch, S*A)` - optimal occupancy measure
- `basis`: `(batch, S)` - basis column indices

#### 2. Sparse LP Solver

Uses Gurobi with COO (Coordinate) sparse format. Efficient for large, sparse problems.

```python
from mdp_lp.solvers import SparseLPSolver, solve_lp_sparse

# Class interface
solver = SparseLPSolver()
y_star, basis = solver.solve(H, r, gamma)

# Function interface
y_star, basis = solve_lp_sparse(H, r, gamma)
```

**Benefits:**
- Memory efficient: Only stores non-zero entries
- Faster for sparse H matrices (common in MDPs)
- Same API as dense solver

#### 3. Differentiable LP Solver

PyTorch autograd function for end-to-end differentiation. Used by DFL trainer.

```python
from mdp_lp.solvers import DifferentiableLPSolver

# Used as autograd Function
y_star = DifferentiableLPSolver.apply(H, r, gamma)
```

**Current implementation:**
- Computes `∂v/∂r = y*` (gradient w.r.t. rewards)
- `∂v/∂H` and `∂v/∂gamma` are placeholders (return zeros)

**Note**: Full gradient computation for H is planned for future versions.

### Trainers

Two training strategies are provided:

#### 1. DFL Trainer (Direct Feedback Learning)

End-to-end differentiable training. Solves LP during forward pass and backpropagates through it.

```python
from mdp_lp.trainers import DFLTrainer

trainer = DFLTrainer(predictor, beta=0.9, optimizer=optimizer)

# Forward pass
v_star, y_star, P, r = trainer.forward(x)

# Training step (supervised)
loss, v_pred = trainer.train_step(x, v_true=target_values)

# Training step (unsupervised - maximize)
loss, v_pred = trainer.train_step(x, v_true=None)
```

**Loss functions:**
- **Supervised**: `loss = MSE(v_star, v_true)` when `v_true` is provided
- **Unsupervised**: `loss = -v_star.mean()` to maximize value

**Use when:**
- You want end-to-end gradients
- You have target values or want to maximize value
- You need differentiable optimization

#### 2. TwoStage Trainer

Two-stage training: first learn P and r, then solve LP at evaluation.

```python
from mdp_lp.trainers import TwoStageTrainer

trainer = TwoStageTrainer(predictor, beta=0.9, optimizer=optimizer)

# Training step (supervised on P, r)
loss = trainer.train_step(x, P_true, r_true)

# Evaluation step (solve LP)
v_star = trainer.eval_step(x)
```

**Loss function:**
- `loss = L1(P_pred, P_true) + L1(r_pred, r_true)`

**Use when:**
- You have ground truth P and r
- You want faster training (no LP solve during training)
- You only need LP solution at evaluation time

**Trade-off:**
- ✅ Faster training (no LP solve)
- ❌ No end-to-end gradients through LP
- ❌ Requires ground truth P and r

## Usage Examples

### Example 1: Compare Dense vs Sparse Solvers

```python
import torch
from mdp_lp.mdp import MDP
from mdp_lp.solvers import solve_lp_dense, solve_lp_sparse

mdp = MDP(beta=0.9)
P = torch.tensor([[...]])  # your MDP
r = torch.tensor([[...]])

H, r_flat, gamma = mdp.lp_formulation(P, r)

# Dense solver
y_dense, basis_dense = solve_lp_dense(H[0], r_flat[0], gamma[0])

# Sparse solver
y_sparse, basis_sparse = solve_lp_sparse(H[0], r_flat[0], gamma[0])

# Should match (within numerical precision)
assert torch.allclose(y_dense, y_sparse, atol=1e-5)
```

### Example 2: Full Training Loop (DFL)

```python
import torch
import torch.nn as nn
from mdp_lp.trainers import DFLTrainer

# Setup
predictor = MyPredictor(input_dim=10, S=5, A=3)
optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
trainer = DFLTrainer(predictor, beta=0.9, optimizer=optimizer)

# Training loop
for epoch in range(100):
    for x, v_true in dataloader:
        loss, v_pred = trainer.train_step(x, v_true=v_true)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}, v_pred: {v_pred.mean().item():.4f}")
```

### Example 3: Two-Stage Training

```python
from mdp_lp.trainers import TwoStageTrainer

trainer = TwoStageTrainer(predictor, beta=0.9, optimizer=optimizer)

# Training phase: learn P and r
for epoch in range(100):
    for x, P_true, r_true in dataloader:
        loss = trainer.train_step(x, P_true, r_true)

# Evaluation phase: solve LP
with torch.no_grad():
    for x in eval_dataloader:
        v_star = trainer.eval_step(x)
        print(f"Optimal value: {v_star.item():.4f}")
```

### Example 4: Custom Predictor Architecture

```python
import torch
import torch.nn as nn
from mdp_lp.trainers import DFLTrainer

class TransformerPredictor(nn.Module):
    """Example: Transformer-based predictor"""
    def __init__(self, input_dim, S, A, d_model=128):
        super().__init__()
        self.encoder = nn.TransformerEncoder(...)
        self.P_head = nn.Linear(d_model, S * A * S)
        self.r_head = nn.Linear(d_model, S * A)
        self.S, self.A = S, A
    
    def forward(self, x):
        # Your architecture here
        encoded = self.encoder(x)
        P_flat = self.P_head(encoded)
        r_flat = self.r_head(encoded)
        # Reshape and normalize...
        return P, r

# Use with trainer
predictor = TransformerPredictor(input_dim=10, S=5, A=3)
trainer = DFLTrainer(predictor, beta=0.9, optimizer=optimizer)
```

## Mathematical Background

### MDP to LP Conversion

For a discounted MDP `(S, A, P, r, β, γ)`, the optimal value function can be found by solving:

**Bellman Equation:**
```
v*(s) = max_a [r(s,a) + β Σ_{s'} P(s'|s,a) v*(s')]
```

**LP Formulation:**
```
maximize:    Σ_{s,a} r(s,a) · y(s,a)
subject to:  Σ_a y(s',a) = γ(s') + β Σ_{s,a} P(s'|s,a) y(s,a)  ∀s'
             y(s,a) ≥ 0  ∀s,a
```

Where `y(s,a)` is the **occupancy measure**: the expected discounted time spent in state `s` taking action `a`.

### Constraint Matrix H

The constraint matrix `H` encodes the flow balance:

```
H[s', (s,a)] = δ[s'=s] - β · P(s'|s,a)
```

This ensures:
- **Identity term** `δ[s'=s]`: Occupancy in state `s'` from being in `s'`
- **Transition term** `-β·P(s'|s,a)`: Discounted flow out of `s'` via transitions

### Optimality

The LP solution `y*` gives:
- **Optimal value**: `v* = Σ_{s,a} r(s,a) · y*(s,a)`
- **Optimal policy**: `π*(s) = argmax_a y*(s,a)` (deterministic)

## API Reference

### MDP

```python
class MDP:
    def __init__(self, beta, gamma=None)
    def build_H(self, P) -> Tensor  # (batch, S, S*A)
    def build_r(self, r) -> Tensor  # (batch, S*A)
    def build_gamma(self, batch, S, device) -> Tensor  # (batch, S)
    def lp_formulation(self, P, r) -> Tuple[Tensor, Tensor, Tensor]
```

### Solvers

```python
# Dense
class DenseLPSolver:
    def solve_single(self, H_i, r_i, gamma_i) -> Tuple[Tensor, Tensor]
    def solve(self, H, r, gamma) -> Tuple[Tensor, Tensor]

def solve_lp_dense(H, r, gamma) -> Tuple[Tensor, Tensor]
def solve_lp_gurobi(H, r, gamma) -> Tensor  # Returns only y_star

# Sparse
class SparseLPSolver:
    def solve_single(self, H_i, r_i, gamma_i) -> Tuple[Tensor, Tensor]
    def solve(self, H, r, gamma) -> Tuple[Tensor, Tensor]

def solve_lp_sparse(H, r, gamma) -> Tuple[Tensor, Tensor]

# Differentiable
class DifferentiableLPSolver(autograd.Function):
    @staticmethod
    def forward(ctx, H, r, gamma) -> Tensor
    @staticmethod
    def backward(ctx, grad_y) -> Tuple[Tensor, Tensor, Tensor]
```

### Trainers

```python
class DFLTrainer:
    def __init__(self, predictor, beta, gamma=None, optimizer=None)
    def forward(self, x) -> Tuple[Tensor, Tensor, Tensor, Tensor]
    def train_step(self, x, v_true=None) -> Tuple[float, Tensor]

class TwoStageTrainer:
    def __init__(self, predictor, beta, gamma=None, optimizer=None)
    def train_step(self, x, P_true, r_true) -> float
    def eval_step(self, x) -> Tensor
```

## Why No Neural Networks in mdp_lp?

`mdp_lp` is designed with **separation of concerns**:

- **`mdp_lp` provides**: MDP/LP infrastructure, solvers, training wrappers
- **You provide**: Neural network architecture (predictor)

This design:
- ✅ **Flexibility**: Use any architecture (MLP, CNN, Transformer, etc.)
- ✅ **Reusability**: Same LP infrastructure works with different predictors
- ✅ **Testability**: Easy to test with simple predictors
- ✅ **Clean API**: Library doesn't assume specific network design

The trainers accept any `nn.Module` that implements:
```python
def forward(self, x) -> Tuple[P, r]:
    # P: (batch, S, A, S) - transition probabilities
    # r: (batch, S, A) - rewards
    return P, r
```

## Performance Considerations

### When to Use Each Solver

- **Dense Solver**: Small problems (< 1000 variables), dense H matrices
- **Sparse Solver**: Large problems (> 1000 variables), sparse H matrices
- **Differentiable Solver**: When you need gradients (used automatically by DFL trainer)

### Memory Usage

- **Dense**: O(S × S×A) memory for H matrix
- **Sparse**: O(nnz) memory, where nnz = number of non-zeros

For large MDPs (S=1000, A=10), sparse solver can be 100× more memory efficient.

## Contributing

This is a research codebase. For questions or issues, please contact the maintainers.

## License

[Specify your license here]

## Citation

If you use this code in your research, please cite:

```bibtex
@software{mdp_lp,
  title = {mdp_lp: MDP Linear Programming Solver},
  author = {[Your Name]},
  year = {2024},
  url = {[Your URL]}
}
```

