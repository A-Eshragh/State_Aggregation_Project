"""Hyperparameters for continuous MDP CartPole experiments."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CartPoleConfig:
    # ── Environment ──
    gamma: float = 0.99
    max_steps: int = 200
    reward_type: str = "cos_theta"  # "cos_theta" or "constant"
    noise_std: float = 0.0          # Gaussian noise on transitions (0 = deterministic)

    # CartPole physics
    m_cart: float = 1.0
    m_pole: float = 0.1
    pole_length: float = 0.5
    gravity: float = 9.8
    force_mag: float = 10.0
    dt: float = 0.02

    # State bounds for normalization
    x_max: float = 2.4
    x_dot_max: float = 3.0
    theta_max: float = 0.2095  # ~12 degrees
    theta_dot_max: float = 3.0

    # ── Features ──
    feature_type: str = "rff"  # "rff" or "neural"
    n_features: int = 128      # d (number of basis functions)
    rff_bandwidth: float = 0.5
    neural_hidden: List[int] = field(default_factory=lambda: [128, 128])

    # ── LP solver ──
    # Pure LP (no QP regularization) — uses basis-restricted sensitivity backward
    ridge: float = 1e-4  # diagonal regularization on M for numerical stability

    # ── Data collection ──
    n_trajectories: int = 500
    buffer_size: int = 50000

    # ── Training (neural / DFL) ──
    lr: float = 1e-3
    n_epochs: int = 100
    n_iterations: int = 20       # outer iterations for iterative refinement
    dfl_lambda: float = 1.0      # penalty weight for flow residual
    exploration_eps: float = 0.1  # epsilon-greedy exploration

    # ── Evaluation ──
    n_eval_episodes: int = 100

    # ── Baselines ──
    vi_bins_per_dim: int = 20    # for discretize + VI
    vi_n_iters: int = 500
    agg_epsilon: float = 0.5     # for adaptive aggregation
