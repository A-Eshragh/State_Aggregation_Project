# Testing Guide for mdp_lp

Complete guide to testing the `mdp_lp` codebase, including all components and trainers.

## Quick Start

### Run All Tests

```bash
python -m mdp_lp.test_all
```

This runs all 13 tests covering:
- Core MDP functionality
- LP solvers (dense, sparse, differentiable)
- Both trainers (DFL and TwoStage)
- Gradient computation

### Run Original Test Suite

```bash
python -m mdp_lp.tests.tests
```

### Run Individual Tests

```python
from mdp_lp.tests.tests import (
    test_H_matrix,
    test_dense_lp_solver,
    test_dfl_trainer_forward,
    # ... etc
)

test_H_matrix()
test_dense_lp_solver()
```

## Test Coverage

### Core MDP Tests (3 tests)

#### TEST 1: H Matrix Construction
- **What**: Verifies constraint matrix H is built correctly
- **Formula**: `H[s', (s,a)] = δ[s'=s] - β·P(s'|s,a)`
- **Checks**: Shape correctness, value correctness
- **Speed**: ⚡ Instant (just matrix math)

#### TEST 2: Reward Flattening
- **What**: Converts `r[batch, S, A]` → `r[batch, S*A]`
- **Checks**: Reshape correctness
- **Speed**: ⚡ Instant (just reshape)

#### TEST 3: LP Formulation
- **What**: Combines H, r, gamma into LP format
- **Checks**: Shape correctness of all components
- **Speed**: ⚡ Instant (shape checks)

### LP Solver Tests (4 tests)

#### TEST 4: Dense LP Solver
- **What**: Solves LP using Gurobi (dense format)
- **Problem**: 2 states, 2 actions = 4 variables, 2 constraints
- **Checks**: Solution correctness, non-negativity
- **Speed**: 🐢 ~10-50ms (Gurobi solve)

#### TEST 5: Gradient Flow
- **What**: Tests gradients through DifferentiableLPSolver
- **Checks**: `∂v/∂r` is computed correctly
- **Speed**: ⚡ ~5-10ms (single backward pass)

#### TEST 12: Sparse LP Solver
- **What**: Solves LP using Gurobi (COO sparse format)
- **Checks**: Matches dense solver results
- **Speed**: 🐢 ~10-50ms (Gurobi solve)

#### TEST 13: Sparse LP Solver (Batched)
- **What**: Tests batched sparse solving
- **Checks**: Matches dense solver for batch
- **Speed**: 🐢 ~20-100ms (multiple solves)

### Trainer Tests (6 tests)

#### TEST 6: DFL Trainer Forward Pass
- **What**: Tests forward pass through DFL trainer
- **Flow**: `x → predictor → P, r → LP → v*`
- **Checks**: Output shapes, finite values
- **Speed**: 🐢 ~10-50ms (forward + LP solve)

#### TEST 7: DFL Trainer Training (with target)
- **What**: Supervised training with target values
- **Loss**: `MSE(v_star, v_true)`
- **Checks**: Loss computation, gradient flow
- **Speed**: 🐢 ~20-100ms (forward + backward + LP)

#### TEST 8: DFL Trainer Training (maximize)
- **What**: Unsupervised training to maximize value
- **Loss**: `-v_star.mean()`
- **Checks**: Loss = -v_pred (within precision)
- **Speed**: 🐢 ~20-100ms (forward + backward + LP)

#### TEST 9: TwoStage Trainer Training
- **What**: Training on P and r directly
- **Loss**: `L1(P_pred, P_true) + L1(r_pred, r_true)`
- **Checks**: Loss computation, gradient flow
- **Speed**: ⚡ ~5-10ms (no LP solve during training!)

#### TEST 10: TwoStage Trainer Evaluation
- **What**: Evaluation step with LP solve
- **Flow**: `x → predictor → P, r → LP → v*`
- **Checks**: Output shape, finite values
- **Speed**: 🐢 ~10-50ms (forward + LP solve)

#### TEST 11: Trainer Gradients
- **What**: Verifies both trainers compute gradients
- **Checks**: Gradients exist after training step
- **Speed**: 🐢 ~30-150ms (2 training steps)

## Why Tests Are Fast

All tests use a **tiny 2-state, 2-action MDP** (only 4 state-action pairs):

| Component | Test Size | Real-World Size | Speedup |
|-----------|-----------|-----------------|---------|
| States (S) | 2 | 100-1000+ | 50-500× |
| Actions (A) | 2 | 10-100+ | 5-50× |
| Variables | 4 | 1,000-100k+ | 250-25k× |
| Constraints | 2 | 100-1000+ | 50-500× |
| Batch size | 1 | 32-256+ | 32-256× |
| Training steps | 1 | 1,000-100k+ | 1,000-100k× |

**Total speedup: ~10⁶ - 10⁹× faster than real training!**

### Test Configuration

- **MDP size**: 2 states × 2 actions = 4 variables
- **H matrix**: 2×4 (8 numbers)
- **Neural network**: 4 inputs → 16 outputs (1 layer)
- **Batch size**: 1
- **Training steps**: 1 (not full epochs)

### Why This Design?

1. **Fast feedback**: Know if code works in <1 second
2. **Catch bugs early**: Shape errors, import errors, etc.
3. **Verify logic**: Math is correct even on tiny examples
4. **CI/CD friendly**: Can run in seconds, not minutes

**Real training would be:**
- 100+ states, 10+ actions
- Batch size 32-256
- 1,000+ epochs
- Minutes to hours per run

But the **logic is the same**, so if tests pass, the code works! 🎯

## Detailed Test Breakdown

### TEST 1: H Matrix Construction

**What it does:**
- Builds constraint matrix H for the LP
- Formula: `H[s', (s,a)] = δ[s'=s] - β·P(s'|s,a)`
- For 2 states, 2 actions: H is 2×4 matrix

**Example:**
```python
P = [[[1,0], [0,1]],    # state 0: a0→s0, a1→s1
     [[0,1], [1,0]]]     # state 1: a0→s1, a1→s0

H = [[0.1, 1.0, 0.0, -0.9],   # constraint for state 0
     [0.0, -0.9, 0.1, 1.0]]   # constraint for state 1
```

**Why fast:**
- Just matrix arithmetic, no optimization
- 2×4 = 8 numbers to compute
- Pure PyTorch operations

### TEST 2: Reward Flattening

**What it does:**
- Converts `r[batch, S, A]` → `r[batch, S*A]`
- Example: `[[1,2], [3,4]]` → `[1,2,3,4]`

**Why fast:**
- Single `.reshape()` call
- No computation, just memory layout

### TEST 3: LP Formulation

**What it does:**
- Combines H, r, gamma into LP format
- Checks shapes are correct

**Why fast:**
- Just calls `build_H()` and `build_r()` (already tested)
- Shape checks only

### TEST 4: Dense LP Solver

**What it does:**
- Solves: `maximize r^T y, subject to H y = γ, y ≥ 0`
- Uses Gurobi to find optimal occupancy measure y*

**What it's solving:**
```
maximize:  [1, 1, 0, 0] · y
subject to: H · y = [0.5, 0.5]
            y ≥ 0
```

**Result:** `y* = [9.5, 0, 0, 0.5]`, `v* ≈ 9.5`

**Why relatively fast:**
- Tiny problem: 4 variables, 2 constraints
- Gurobi is highly optimized
- Single LP solve (not batched)

### TEST 5: Gradient Flow

**What it does:**
- Tests that gradients flow through DifferentiableLPSolver
- Uses autograd to compute `∂v/∂r`

**What happens:**
```python
v = sum(r_flat * y_star)  # value function
v.backward()               # compute gradients
# r.grad exists ✓
```

**Why fast:**
- Single backward pass
- DifferentiableLP only computes `grad_r` (not `grad_H` yet)
- No actual optimization, just gradient computation

### TEST 6: DFL Trainer Forward

**What it does:**
- Creates tiny neural network (4→16)
- Runs forward pass: `x → predictor → P, r → LP → v*`
- Checks output shapes

**Flow:**
```
x (1,4) → SimplePredictor → P (1,2,2,2), r (1,2,2)
  → MDP.lp_formulation() → H, r_flat, gamma
  → DifferentiableLPSolver → y* (1,4)
  → v* = sum(r_flat * y*) → v* (1,)
```

**Why fast:**
- Tiny network: 4 inputs, 16 outputs (only 1 layer!)
- Single forward pass
- One LP solve (via DifferentiableLPSolver)

### TEST 7: DFL Trainer Training (with target)

**What it does:**
- Supervised learning: minimize `MSE(v_pred, v_true)`
- One training step: forward → loss → backward → optimizer.step()

**Loss:**
```python
loss = MSE(v_star, v_true)  # e.g., MSE(14.02, 5.0) = 81.38
loss.backward()             # compute gradients
optimizer.step()            # update weights (tiny update)
```

**Why fast:**
- Single training step (not an epoch!)
- Tiny network (4→16)
- One LP solve + one backward pass

### TEST 8: DFL Trainer Training (maximize)

**What it does:**
- Unsupervised: maximize v* by minimizing -v*
- Loss = `-v_star.mean()`

**Loss:**
```python
loss = -v_star.mean()  # e.g., -(-1.40) = 1.40
# Minimizing this maximizes v_star
```

**Why fast:**
- Same as Test 7, just different loss function
- One step only

### TEST 9: TwoStage Trainer Training

**What it does:**
- Supervised learning on P and r directly
- Loss = `L1(P_pred, P_true) + L1(r_pred, r_true)`
- **No LP solve during training!**

**Loss:**
```python
P_pred, r_pred = predictor(x)
loss = L1(P_pred, P_true) + L1(r_pred, r_true)
loss.backward()
```

**Why very fast:**
- No LP solving (that's the "two-stage" part!)
- Just neural network forward + backward
- Tiny network (4→16)

### TEST 10: TwoStage Trainer Eval

**What it does:**
- Evaluation: solve LP with predicted P, r
- Returns optimal value v*

**Why fast:**
- Single forward pass (no gradients)
- One LP solve
- Tiny problem (2×4)

### TEST 11: Trainer Gradients

**What it does:**
- Verifies both trainers compute gradients correctly
- Runs one training step for each trainer

**Why fast:**
- Just Test 7 + Test 9 combined
- Two small training steps

### TEST 12: Sparse LP Solver

**What it does:**
- Solves LP using sparse COO format
- Compares with dense solver to verify correctness

**Checks:**
- Solutions match dense solver (within 1e-5 tolerance)
- Non-negativity constraints satisfied
- Correct output shapes

### TEST 13: Sparse LP Solver (Batched)

**What it does:**
- Tests batched sparse solving
- Compares with batched dense solver

**Checks:**
- Batch processing works correctly
- Solutions match dense solver for all batch elements

## Test Utilities

### SimplePredictor

A minimal neural network used for testing:

```python
class SimplePredictor(nn.Module):
    def __init__(self, input_dim=4, S=2, A=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, S * A * S + S * A)
        self.S = S
        self.A = A
    
    def forward(self, x):
        # Outputs P and r from input features
        return P, r
```

**Purpose:**
- Provides a simple predictor for trainer tests
- Not meant for real use (too simple)
- Demonstrates the predictor interface

## Running Tests

### Command Line

```bash
# Run all tests
python -m mdp_lp.test_all

# Run original test suite
python -m mdp_lp.tests.tests
```

### Python Script

```python
from mdp_lp.tests.tests import (
    test_H_matrix,
    test_dense_lp_solver,
    test_sparse_lp_solver,
    test_dfl_trainer_forward,
    # ... etc
)

# Run specific tests
test_H_matrix()
test_dense_lp_solver()
```

### Expected Output

```
======================================================================
RUNNING ALL MDP_LP TESTS
======================================================================

======================================================================
Core MDP Tests
======================================================================
✓ H shape correct
✓ H values correct
✓ r flattening correct
✓ LP formulation shapes correct

======================================================================
LP Solver Tests
======================================================================
✓ LP solver correct
✓ Gradients flow through DifferentiableLP
✓ Sparse LP solver correct and matches dense solver
✓ Sparse LP solver (batched) works and matches dense solver

======================================================================
Trainer Tests
======================================================================
✓ DFL Trainer forward pass works
✓ DFL Trainer train step (with target) works
✓ DFL Trainer train step (maximize) works
✓ TwoStage Trainer train step works
✓ TwoStage Trainer eval step works
✓ Both trainers compute gradients correctly

======================================================================
TEST SUMMARY
======================================================================
Passed: 13
Failed: 0
Total: 13

======================================================================
ALL TESTS PASSED ✓
======================================================================
```

## Notes

- **Gurobi license messages**: May appear (this is normal for academic licenses)
- **Test MDP**: All tests use 2-state, 2-action MDP for speed
- **SimplePredictor**: Used for trainer tests (not for real use)
- **Numerical precision**: Tests use `atol=1e-5` for floating point comparisons

## Troubleshooting

### Tests Fail with Gurobi Error

**Problem**: `GurobiError: No Gurobi license found`

**Solution**: 
1. Get academic license from [Gurobi](https://www.gurobi.com/academia/academic-program-and-licenses/)
2. Install license: `grbgetkey <license-key>`

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'mdp_lp'`

**Solution**: 
```bash
# Make sure you're in the project root
cd /path/to/State_Aggregation_Project
python -m mdp_lp.test_all
```

### Numerical Precision Issues

**Problem**: Tests fail with small numerical differences

**Solution**: This is normal. Tests use `atol=1e-5`. If failures persist, check:
- Gurobi version compatibility
- PyTorch version compatibility
- Floating point precision settings

## Adding New Tests

To add a new test:

1. Add test function to `mdp_lp/tests/tests.py`:
```python
def test_my_feature():
    print_section("TEST N: MY FEATURE")
    # Your test code
    assert condition, "Error message"
    print("✓ My feature works")
```

2. Add to `test_all.py`:
```python
from mdp_lp.tests.tests import test_my_feature

tests = [
    ...
    ("My Category", [test_my_feature]),
]
```

3. Run tests to verify:
```bash
python -m mdp_lp.test_all
```
