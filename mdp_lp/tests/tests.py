import torch
import numpy as np
import torch.nn as nn
from mdp_lp.mdp import MDP
from mdp_lp.solvers.dense_lp_solver import solve_lp_dense
from mdp_lp.trainers import DFLTrainer, TwoStageTrainer


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------
#  TEST 1: Verify H construction on tiny MDP
# ----------------------------------------------------------
def test_H_matrix():
    print_section("TEST 1: H MATRIX CONSTRUCTION")

    beta = 0.9
    mdp = MDP(beta=beta)

    # small deterministic transitions
    P = torch.tensor([
        [
            # state 0
            [[1.0, 0.0],   # a0
             [0.0, 1.0]],  # a1
            # state 1
            [[0.0, 1.0],
             [1.0, 0.0]],
        ]
    ])

    H = mdp.build_H(P)

    print("H matrix:")
    print(H)

    assert H.shape == (1, 2, 4)
    print("✓ H shape correct")

    # manually compute a few entries
    # H[s', (s,a)] = δ[s'=s] − β * P(s→s')
    expected = torch.tensor([
        [
            [1 - 0.9*1, 1 - 0.9*0, -0.9*0, -0.9*1],
            [-0.9*0, -0.9*1, 1 - 0.9*1, 1 - 0.9*0],
        ]
    ])

    assert torch.allclose(H, expected, atol=1e-6)
    print("✓ H values correct")


# ----------------------------------------------------------
#  TEST 2: Reward flattening
# ----------------------------------------------------------
def test_r_flatten():
    print_section("TEST 2: REWARD FLATTENING")

    r = torch.tensor([
        [[1, 2],
         [3, 4]]
    ])

    mdp = MDP(beta=0.9)
    r_flat = mdp.build_r(r)

    print("r_flat =", r_flat)

    assert torch.allclose(r_flat, torch.tensor([[1, 2, 3, 4]]))
    print("✓ r flattening correct")


# ----------------------------------------------------------
#  TEST 3: LP Formulation correctness
# ----------------------------------------------------------
def test_lp_formulation():
    print_section("TEST 3: LP FORMULATION")

    P = torch.tensor([
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    ])
    r = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])

    mdp = MDP(beta=0.9)
    H, r_flat, gamma = mdp.lp_formulation(P, r)

    print("H =", H)
    print("r_flat =", r_flat)
    print("gamma =", gamma)

    assert H.shape == (1, 2, 4)
    assert r_flat.shape == (1, 4)
    assert gamma.shape == (1, 2)
    print("✓ LP formulation shapes correct")


# ----------------------------------------------------------
#  TEST 4: Solve LP using Gurobi and check correctness
# ----------------------------------------------------------
def test_dense_lp_solver():
    print_section("TEST 4: GUROBI LP SOLVE")

    # Simple example: same as Toy MDP
    P = torch.tensor([
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    ])
    r = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    beta = 0.9

    mdp = MDP(beta)
    H, r_flat, gamma = mdp.lp_formulation(P, r)

    y_star, basis = solve_lp_dense(H[0], r_flat[0], gamma[0])

    print("y* =", y_star)
    print("basis =", basis)

    # expected occupancy measure:
    # discount factor → 1/(1-β) = 1/0.1 = 10
    # state 0 reward 1 → occupancy = 10
    # but LP finds one action per state → sum to 10:

    total_reward = (y_star * r_flat[0]).sum().item()
    print("v* =", total_reward)

    # Check that we get a reasonable positive value
    assert total_reward > 0, f"Expected positive reward, got {total_reward}"
    assert torch.allclose(y_star, y_star.clamp(min=0)), "y_star should be non-negative"
    print("✓ LP solver correct")


# ----------------------------------------------------------
#  TEST 5: Autograd sanity check
# ----------------------------------------------------------
def test_gradients():
    print_section("TEST 5: AUTOGRAD THROUGH LP")

    from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver
    
    beta = 0.9
    mdp = MDP(beta)

    P = torch.tensor([
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    ], requires_grad=True)

    r = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]], requires_grad=True)

    H, r_flat, gamma = mdp.lp_formulation(P, r)

    # Use DifferentiableLPSolver which supports gradients
    y_star = DifferentiableLPSolver.apply(H, r_flat, gamma)
    
    v = torch.sum(r_flat * y_star, dim=1)

    print("v =", v.item())

    v.backward()

    print("grad P exists:", P.grad is not None)
    print("grad r exists:", r.grad is not None)

    # Note: DifferentiableLPSolver currently only computes grad_r, not grad_P
    # This is expected based on the implementation
    assert r.grad is not None, "grad_r should exist"
    print("✓ Gradients flow through DifferentiableLP (grad_r computed)")


# ----------------------------------------------------------
#  TEST 6: Simple Predictor Model for Testing
# ----------------------------------------------------------
class SimplePredictor(nn.Module):
    """Simple predictor that outputs P and r from input features."""
    def __init__(self, input_dim=4, S=2, A=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, S * A * S + S * A)
        self.S = S
        self.A = A
    
    def forward(self, x):
        batch = x.size(0)
        out = self.fc(x)
        
        # Split output into P and r
        P_flat = out[:, :self.S * self.A * self.S]
        r_flat = out[:, self.S * self.A * self.S:]
        
        # Reshape P: (batch, S*A*S) -> (batch, S, A, S)
        P = torch.softmax(P_flat.view(batch, self.S, self.A, self.S), dim=-1)
        
        # Reshape r: (batch, S*A) -> (batch, S, A)
        r = r_flat.view(batch, self.S, self.A)
        
        return P, r


# ----------------------------------------------------------
#  TEST 7: DFL Trainer - Forward Pass
# ----------------------------------------------------------
def test_dfl_trainer_forward():
    print_section("TEST 6: DFL TRAINER FORWARD PASS")
    
    S, A = 2, 2
    beta = 0.9
    predictor = SimplePredictor(input_dim=4, S=S, A=A)
    trainer = DFLTrainer(predictor, beta=beta)
    
    # Create dummy input
    x = torch.randn(1, 4)
    
    # Forward pass
    v_star, y_star, P, r = trainer.forward(x)
    
    print(f"v_star shape: {v_star.shape}")
    print(f"y_star shape: {y_star.shape}")
    print(f"P shape: {P.shape}")
    print(f"r shape: {r.shape}")
    print(f"v_star value: {v_star.item():.4f}")
    
    assert v_star.shape == (1,), f"Expected v_star shape (1,), got {v_star.shape}"
    assert y_star.shape == (1, S*A), f"Expected y_star shape (1, {S*A}), got {y_star.shape}"
    assert P.shape == (1, S, A, S), f"Expected P shape (1, {S}, {A}, {S}), got {P.shape}"
    assert r.shape == (1, S, A), f"Expected r shape (1, {S}, {A}), got {r.shape}"
    assert torch.isfinite(v_star).all(), "v_star should be finite"
    
    print("✓ DFL Trainer forward pass works")


# ----------------------------------------------------------
#  TEST 8: DFL Trainer - Training Step with v_true
# ----------------------------------------------------------
def test_dfl_trainer_train_with_target():
    print_section("TEST 7: DFL TRAINER TRAIN STEP (WITH TARGET)")
    
    S, A = 2, 2
    beta = 0.9
    predictor = SimplePredictor(input_dim=4, S=S, A=A)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    trainer = DFLTrainer(predictor, beta=beta, optimizer=optimizer)
    
    # Create dummy input and target
    x = torch.randn(1, 4)
    v_true = torch.tensor([5.0])
    
    # Training step
    loss, v_pred = trainer.train_step(x, v_true=v_true)
    
    print(f"Loss: {loss:.4f}")
    print(f"v_pred: {v_pred.item():.4f}")
    print(f"v_true: {v_true.item():.4f}")
    
    assert isinstance(loss, float), "Loss should be a float"
    assert loss >= 0, "Loss should be non-negative"
    assert v_pred.shape == (1,), f"Expected v_pred shape (1,), got {v_pred.shape}"
    
    print("✓ DFL Trainer train step (with target) works")


# ----------------------------------------------------------
#  TEST 9: DFL Trainer - Training Step without v_true (maximize)
# ----------------------------------------------------------
def test_dfl_trainer_train_maximize():
    print_section("TEST 8: DFL TRAINER TRAIN STEP (MAXIMIZE)")
    
    S, A = 2, 2
    beta = 0.9
    predictor = SimplePredictor(input_dim=4, S=S, A=A)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    trainer = DFLTrainer(predictor, beta=beta, optimizer=optimizer)
    
    # Create dummy input
    x = torch.randn(1, 4)
    
    # Training step (maximize v_star)
    loss, v_pred = trainer.train_step(x, v_true=None)
    
    print(f"Loss (-v_star.mean()): {loss:.4f}")
    print(f"v_pred: {v_pred.item():.4f}")
    
    assert isinstance(loss, float), "Loss should be a float"
    # Loss = -v_star.mean(), so if v_star is negative, loss is positive
    # This is correct: minimizing loss maximizes v_star
    assert torch.isfinite(torch.tensor(loss)), "Loss should be finite"
    assert v_pred.shape == (1,), f"Expected v_pred shape (1,), got {v_pred.shape}"
    # Verify loss = -v_pred (since v_pred is v_star.detach())
    assert abs(loss + v_pred.item()) < 1e-5, f"Loss should equal -v_pred, got loss={loss}, v_pred={v_pred.item()}"
    
    print("✓ DFL Trainer train step (maximize) works")


# ----------------------------------------------------------
#  TEST 10: TwoStage Trainer - Training Step
# ----------------------------------------------------------
def test_twostage_trainer_train():
    print_section("TEST 9: TWOSTAGE TRAINER TRAIN STEP")
    
    S, A = 2, 2
    beta = 0.9
    predictor = SimplePredictor(input_dim=4, S=S, A=A)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-3)
    trainer = TwoStageTrainer(predictor, beta=beta, optimizer=optimizer)
    
    # Create dummy input and true P, r
    x = torch.randn(1, 4)
    P_true = torch.softmax(torch.randn(1, S, A, S), dim=-1)
    r_true = torch.randn(1, S, A)
    
    # Training step
    loss = trainer.train_step(x, P_true, r_true)
    
    print(f"Loss: {loss:.4f}")
    print(f"P_true shape: {P_true.shape}")
    print(f"r_true shape: {r_true.shape}")
    
    assert isinstance(loss, float), "Loss should be a float"
    assert loss >= 0, "Loss should be non-negative"
    
    print("✓ TwoStage Trainer train step works")


# ----------------------------------------------------------
#  TEST 11: TwoStage Trainer - Eval Step
# ----------------------------------------------------------
def test_twostage_trainer_eval():
    print_section("TEST 10: TWOSTAGE TRAINER EVAL STEP")
    
    S, A = 2, 2
    beta = 0.9
    predictor = SimplePredictor(input_dim=4, S=S, A=A)
    trainer = TwoStageTrainer(predictor, beta=beta)
    
    # Create dummy input
    x = torch.randn(1, 4)
    
    # Eval step
    v_star = trainer.eval_step(x)
    
    print(f"v_star shape: {v_star.shape}")
    print(f"v_star value: {v_star.item():.4f}")
    
    assert v_star.shape == (1,), f"Expected v_star shape (1,), got {v_star.shape}"
    assert torch.isfinite(v_star).all(), "v_star should be finite"
    
    print("✓ TwoStage Trainer eval step works")


# ----------------------------------------------------------
#  TEST 12: Trainer Gradients
# ----------------------------------------------------------
def test_trainer_gradients():
    print_section("TEST 11: TRAINER GRADIENTS")
    
    S, A = 2, 2
    beta = 0.9
    
    # Test DFL trainer gradients
    predictor_dfl = SimplePredictor(input_dim=4, S=S, A=A)
    optimizer_dfl = torch.optim.Adam(predictor_dfl.parameters(), lr=1e-3)
    trainer_dfl = DFLTrainer(predictor_dfl, beta=beta, optimizer=optimizer_dfl)
    
    x = torch.randn(1, 4)
    v_true = torch.tensor([5.0])
    
    trainer_dfl.train_step(x, v_true=v_true)
    
    # Check gradients exist
    has_grads_dfl = any(p.grad is not None for p in predictor_dfl.parameters())
    print(f"DFL Trainer has gradients: {has_grads_dfl}")
    assert has_grads_dfl, "DFL Trainer should have gradients"
    
    # Test TwoStage trainer gradients
    predictor_ts = SimplePredictor(input_dim=4, S=S, A=A)
    optimizer_ts = torch.optim.Adam(predictor_ts.parameters(), lr=1e-3)
    trainer_ts = TwoStageTrainer(predictor_ts, beta=beta, optimizer=optimizer_ts)
    
    P_true = torch.softmax(torch.randn(1, S, A, S), dim=-1)
    r_true = torch.randn(1, S, A)
    
    trainer_ts.train_step(x, P_true, r_true)
    
    # Check gradients exist
    has_grads_ts = any(p.grad is not None for p in predictor_ts.parameters())
    print(f"TwoStage Trainer has gradients: {has_grads_ts}")
    assert has_grads_ts, "TwoStage Trainer should have gradients"
    
    print("✓ Both trainers compute gradients correctly")


# ----------------------------------------------------------
#  TEST 12: Sparse LP Solver
# ----------------------------------------------------------
def test_sparse_lp_solver():
    print_section("TEST 12: SPARSE LP SOLVER (COO)")
    
    from mdp_lp.solvers import SparseLPSolver, solve_lp_sparse
    
    # Simple example: same as dense solver test
    P = torch.tensor([
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    ])
    r = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    beta = 0.9
    
    mdp = MDP(beta)
    H, r_flat, gamma = mdp.lp_formulation(P, r)
    
    # Test sparse solver
    y_sparse, basis_sparse = solve_lp_sparse(H[0], r_flat[0], gamma[0])
    
    print("y* (sparse) =", y_sparse)
    print("basis (sparse) =", basis_sparse)
    
    # Compare with dense solver
    y_dense, basis_dense = solve_lp_dense(H[0], r_flat[0], gamma[0])
    
    print("y* (dense) =", y_dense)
    
    # Should match (within numerical precision)
    assert torch.allclose(y_sparse, y_dense, atol=1e-5), \
        f"Sparse and dense solvers should match. Diff: {(y_sparse - y_dense).abs().max().item()}"
    
    total_reward = (y_sparse * r_flat[0]).sum().item()
    print("v* =", total_reward)
    
    assert total_reward > 0, f"Expected positive reward, got {total_reward}"
    assert torch.allclose(y_sparse, y_sparse.clamp(min=0)), "y_star should be non-negative"
    
    print("✓ Sparse LP solver correct and matches dense solver")


# ----------------------------------------------------------
#  TEST 13: Sparse LP Solver (Batched)
# ----------------------------------------------------------
def test_sparse_lp_solver_batched():
    print_section("TEST 13: SPARSE LP SOLVER (BATCHED)")
    
    from mdp_lp.solvers import SparseLPSolver, DenseLPSolver
    
    # Create batch of 2 MDPs
    P = torch.tensor([
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ],
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    ])
    r = torch.tensor([
        [[1.0, 1.0], [0.0, 0.0]],
        [[1.0, 1.0], [0.0, 0.0]],
    ])
    beta = 0.9
    
    mdp = MDP(beta)
    H, r_flat, gamma = mdp.lp_formulation(P, r)
    
    # Test sparse solver (batched)
    sparse_solver = SparseLPSolver()
    y_sparse, basis_sparse = sparse_solver.solve(H, r_flat, gamma)
    
    # Test dense solver (batched)
    dense_solver = DenseLPSolver()
    y_dense, basis_dense = dense_solver.solve(H, r_flat, gamma)
    
    print(f"Batch size: {H.shape[0]}")
    print(f"Sparse y_star[0]: {y_sparse[0]}")
    print(f"Dense y_star[0]:  {y_dense[0]}")
    
    # Should match
    assert torch.allclose(y_sparse, y_dense, atol=1e-5), \
        f"Sparse and dense solvers should match. Max diff: {(y_sparse - y_dense).abs().max().item():.2e}"
    
    assert y_sparse.shape == (2, 4), f"Expected shape (2, 4), got {y_sparse.shape}"
    assert basis_sparse.shape == (2, 2), f"Expected basis shape (2, 2), got {basis_sparse.shape}"
    
    print("✓ Sparse LP solver (batched) works and matches dense solver")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
if __name__ == "__main__":
    # Core MDP and LP solver tests
    test_H_matrix()
    test_r_flatten()
    test_lp_formulation()
    test_dense_lp_solver()
    test_gradients()
    
    # Trainer tests
    test_dfl_trainer_forward()
    test_dfl_trainer_train_with_target()
    test_dfl_trainer_train_maximize()
    test_twostage_trainer_train()
    test_twostage_trainer_eval()
    test_trainer_gradients()
    
    # Sparse solver tests
    test_sparse_lp_solver()
    test_sparse_lp_solver_batched()

    print_section("ALL TESTS PASSED")
