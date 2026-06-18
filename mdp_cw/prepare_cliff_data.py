"""
Ground-truth data generation for the cliff-walking experiments.

Direction:  regimes z  ->  (slip, P_true, r_true)  and  z  ->  w

We store per-sample tensors:
    w:            (B, feat_dim)           observed weather features
    z:            (B,) long                latent regime id (kept for diagnostics)
    slip_compact: (B, n_slots, A, K)       ground-truth compact slip (ambient + overrides)
                                            n_slots = 1 for v1/v2,
                                            n_slots = 1 + |volatile| for v3.
    P_true:       (B, S, A, S)             ground-truth transitions (built from full slip)
    r_true:       (B, S, A)                ground-truth rewards
    y_true:       (B, SA)                  optimal occupancy (for the oracle)
    v_true:       (B,)                     optimal value

Train/test split is a 50/50 random split by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from mdp_cw.cliff_mdp import CliffWalkingLP, set_cliff_penalty
from mdp_cw.weather_dynamics import (
    FeatureGenConfig,
    ground_truth_mdp_tensors,
    sample_features,
    sample_regimes,
    set_regime_set,
)
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver
from mdp_lp.utils import solve_lp


@dataclass
class CliffDataConfig:
    n_samples: int = 200
    feat_dim: int = 8
    noise_std: float = 0.6
    beta: float = 0.95
    train_frac: float = 0.5
    batch_size: int = 32
    seed: int = 0
    regime_weights: Optional[list[float]] = None
    regime_set: str = "v1"                    # "v1" (4 regimes), "v2" (8 policy-diverse), or "v3" (state-dependent)
    center_scale: float = 2.0                 # cluster-center radius (smaller = more overlap)
    ambiguous_pairs: Optional[list[tuple[int, int]]] = None  # e.g. [(3, 4)] in v2
    ambiguous_gap: float = 0.15              # offset between pinned-center ambiguous pairs (0 = identical centers)
    cliff_penalty: float = -100.0             # gym default = -100; use smaller magnitude for soft-cliff variant
    test_regime_weights: Optional[list[float]] = None


def _solve_oracle(P: torch.Tensor, r: torch.Tensor, beta: float):
    """Solve the LP for every sample. Returns y_true (B, SA) and v_true (B,)."""
    solver = DenseLPSolver()

    def lp_solver(H, rr, gamma):
        y, basis, dual = solver.solve(H, rr, gamma)
        return y, basis, dual

    y_list, v_list = [], []
    for i in range(P.shape[0]):
        mdp = CliffWalkingLP(P=P[i:i+1], r=r[i:i+1], beta=beta)
        y_i, obj_i, _, _ = solve_lp(mdp, lp_solver)
        y_list.append(y_i.squeeze(0))
        v_list.append(obj_i.squeeze(0))
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[Data Prep] Solved oracle LP {i+1}/{P.shape[0]}")
    y_true = torch.stack(y_list, dim=0)
    v_true = torch.stack(v_list, dim=0)
    return y_true, v_true


def prepare_cliff_data(cfg: CliffDataConfig):
    """
    Generate the ground-truth dataset and return (train_loader, test_loader, meta).

    meta is a dict with extras (e.g. per-sample ground-truth tensors, feat_cfg).
    """
    torch.manual_seed(cfg.seed)

    set_regime_set(cfg.regime_set)
    set_cliff_penalty(cfg.cliff_penalty)
    print(f"[Data Prep] Regime set = {cfg.regime_set}, cliff_penalty = {cfg.cliff_penalty}")

    n_train = max(1, int(cfg.train_frac * cfg.n_samples))
    n_test = cfg.n_samples - n_train

    print(f"[Data Prep] Sampling {cfg.n_samples} regimes / weather features...")
    if cfg.test_regime_weights is None:
        z = sample_regimes(cfg.n_samples, seed=cfg.seed, weights=cfg.regime_weights)  # (B,)
    else:
        z_train = sample_regimes(n_train, seed=cfg.seed, weights=cfg.regime_weights)
        z_test = sample_regimes(n_test, seed=cfg.seed + 100003, weights=cfg.test_regime_weights)
        z = torch.cat([z_train, z_test], dim=0)

    feat_cfg = FeatureGenConfig(
        feat_dim=cfg.feat_dim, noise_std=cfg.noise_std, seed=cfg.seed + 1,
        center_scale=cfg.center_scale, ambiguous_pairs=cfg.ambiguous_pairs,
        ambiguous_gap=cfg.ambiguous_gap,
    )
    w = sample_features(z, feat_cfg)  # (B, F)

    print("[Data Prep] Building ground-truth slip distributions and MDP tensors...")
    slip_compact, P_true, r_true = ground_truth_mdp_tensors(z)   # (B,slots,A,K), (B,S,A,S), (B,S,A)
    print(f"[Data Prep] slip_compact shape = {tuple(slip_compact.shape)} "
          f"(n_slots={slip_compact.shape[1]})")

    print("[Data Prep] Solving the oracle LP for every sample...")
    y_true, v_true = _solve_oracle(P_true, r_true, cfg.beta)

    dataset = TensorDataset(w, z, slip_compact, P_true, r_true, y_true, v_true)

    if cfg.test_regime_weights is not None:
        if n_test == 0:
            train_ds, test_ds = dataset, dataset
        else:
            train_ds = torch.utils.data.Subset(dataset, range(0, n_train))
            test_ds = torch.utils.data.Subset(dataset, range(n_train, cfg.n_samples))
    else:
        # Deterministic split: same seed always yields the same split.
        gen = torch.Generator().manual_seed(cfg.seed + 7)
        if n_test == 0:
            train_ds, test_ds = dataset, dataset
        else:
            train_ds, test_ds = random_split(dataset, [n_train, n_test], generator=gen)

    bs = min(cfg.batch_size, len(train_ds)) if len(train_ds) else 1
    test_bs = min(cfg.batch_size, len(test_ds)) if len(test_ds) else 1
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=test_bs, shuffle=False)

    from mdp_cw.weather_dynamics import volatile_cells, is_state_dependent

    meta = {
        "feat_cfg": feat_cfg,
        "beta": cfg.beta,
        "n_train": n_train,
        "n_test": n_test,
        "mean_oracle_value": float(v_true.mean().item()),
        "n_slots": int(slip_compact.shape[1]),
        "volatile_cells": list(volatile_cells()),
        "state_dependent": bool(is_state_dependent()),
        "regime_set": cfg.regime_set,
        "cliff_penalty": float(cfg.cliff_penalty),
        "regime_weights": cfg.regime_weights,
        "test_regime_weights": cfg.test_regime_weights,
    }
    print(f"[Data]    n_train={n_train}  n_test={n_test}  oracle_mean={meta['mean_oracle_value']:.3f}")
    return train_loader, test_loader, meta


__all__ = ["CliffDataConfig", "prepare_cliff_data"]
