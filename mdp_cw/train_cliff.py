"""
Training / evaluation loops for the cliff-walking experiments.

Two stages:

    1) Pretraining of the FeatureNet on (w -> slip_true). This is a
       standard supervised objective (KL / CE per action row).  After this
       stage the FeatureNet is FROZEN and only its internal features `h`
       are used downstream.

    2) Main training of the learnable TransitionHead (features -> slip
       distribution) with one of:
           * 'twostage'   — supervised CE/KL on slip distribution (no LP).
           * 'dfl'        — differentiable LP (fixed-basis sensitivity).
           * 'dfl_qpth'   — QP-regularized LP via qpth.
           * 'dfl_ssa'    — DFL on top of a learned soft state aggregation
                            (clusters S=48 into M < S clusters before LP).
           * 'dfl_sketch' — sketched LP: average of K random row-reduced LPs
                            (sketch_ratio controls the row fraction per sketch).

Metrics tracked per epoch:
    * train_reward_per_epoch  : value on the true MDP of the predicted policy.
    * train_pred_loss         : CE(slip_dist_pred, slip_dist_true) (mean).
    * train_dfl_loss          : full DFL loss (only for dfl*).
    * sa_time / solver_time    : timing hooks (aggregation reserved for later).

Evaluation:
    * test_obj_values (predicted reward on true MDP)
    * test_oracle_values
    * test_regret
    * test_slip_ce
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from qpth.qp import QPFunction, QPSolvers

from mdp_cw.cliff_mdp import CLIFF_STATES, CliffWalkingLP, N_ACTIONS, N_COLS, N_SLIP, state_idx
from mdp_cw.features_model import CliffTransitionModel, FeatureNet, TransitionHead
from mdp_cw.weather_dynamics import (
    REGIME_NAMES_V1,
    REGIME_NAMES_V2,
    REGIME_NAMES_V3,
    REGIME_NAMES_V4,
    REGIME_NAMES_V5,
    REGIME_NAMES_V6,
)
from mdp_lp.mdp import MDP as GenericMDP
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver
from mdp_lp.solvers.qpth_qp_solver import solve_occupancy_qp_qpth
from mdp_lp.solvers.sketched_lp import sketched_lp
from mdp_lp.soft_state_aggregation import SoftAggDense
from mdp_lp.utils import solve_lp


def _regime_display_names(meta: dict) -> list[str]:
    rs = str(meta.get("regime_set", "v1")).lower()
    if rs == "v1":
        return list(REGIME_NAMES_V1)
    if rs == "v2":
        return list(REGIME_NAMES_V2)
    if rs == "v3":
        return list(REGIME_NAMES_V3)
    if rs == "v4":
        return list(REGIME_NAMES_V4)
    if rs == "v5":
        return list(REGIME_NAMES_V5)
    if rs == "v6":
        return list(REGIME_NAMES_V6)
    return [f"regime_{i}" for i in range(8)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slip_ce(slip_pred: torch.Tensor, slip_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Mean cross-entropy over K-way categoricals.

    Both tensors must have the same shape ending in K (the slip-outcome
    dimension). All leading dims are averaged. Works for:
        * (B, A, K)                  — state-independent (legacy FeatureNet slip)
        * (B, n_slots, A, K)         — compact slip (v3-aware)
    """
    if slip_pred.shape != slip_true.shape:
        raise ValueError(
            f"slip_pred {tuple(slip_pred.shape)} and slip_true "
            f"{tuple(slip_true.shape)} must match"
        )
    return -(slip_true * torch.log(slip_pred.clamp_min(eps))).sum(dim=-1).mean()


def _slip_ce_per_sample(slip_pred: torch.Tensor, slip_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-sample CE averaged over all compact slots and actions."""
    if slip_pred.shape != slip_true.shape:
        raise ValueError(
            f"slip_pred {tuple(slip_pred.shape)} and slip_true "
            f"{tuple(slip_true.shape)} must match"
        )
    flat = -(slip_true * torch.log(slip_pred.clamp_min(eps))).sum(dim=-1)
    return flat.view(flat.shape[0], -1).mean(dim=1)


def _materialize_dataset(dataset) -> tuple[torch.Tensor, ...]:
    """Stack a map-style dataset into tensors so we can reweight / replay it."""
    rows = [dataset[i] for i in range(len(dataset))]
    cols = list(zip(*rows))
    out = []
    for col in cols:
        first = col[0]
        if torch.is_tensor(first):
            out.append(torch.stack([x.detach().clone() for x in col], dim=0))
        else:
            out.append(list(col))
    return tuple(out)


def _cw_row2_mask(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """States on the cliff-adjacent row, used as the hinge subset."""
    mask = torch.zeros((48,), device=device, dtype=dtype)
    for c in range(N_COLS):
        mask[state_idx(2, c)] = 1.0
    return mask


def _batched_oracle_q(P: torch.Tensor, r: torch.Tensor, beta: float,
                      max_iter: int = 200, tol: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch discounted value iteration for the true MDP."""
    B, S, A, _ = P.shape
    v = torch.zeros(B, S, device=P.device, dtype=P.dtype)
    q = r
    for _ in range(max_iter):
        q = r + beta * torch.einsum("bsan,bn->bsa", P, v)
        v_new = q.max(dim=-1).values
        if torch.max(torch.abs(v_new - v)).item() <= tol:
            v = v_new
            break
        v = v_new
    q = r + beta * torch.einsum("bsan,bn->bsa", P, v)
    return v, q


def _state_action_gap(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (best_action, top1_minus_top2_gap) for q with shape (B,S,A)."""
    top2 = torch.topk(q, k=min(2, q.shape[-1]), dim=-1).values
    best = q.argmax(dim=-1)
    if top2.shape[-1] < 2:
        gap = torch.zeros_like(best, dtype=q.dtype)
    else:
        gap = top2[..., 0] - top2[..., 1]
    return best, gap


def _hinge_scores_from_oracle_q(q_true: torch.Tensor, hinge_mask: torch.Tensor,
                                gap_temp: float = 0.25) -> torch.Tensor:
    """Sample-level hinge score from oracle Q gaps on the row-2 hinge states."""
    _best, gap = _state_action_gap(q_true)
    score = torch.exp(-gap / max(gap_temp, 1e-6)) * hinge_mask.view(1, -1)
    denom = hinge_mask.sum().clamp_min(1.0)
    return score.sum(dim=1) / denom


def _score_train_regret(
    args,
    model,
    train_eval_loader,
    lp_solver,
    qp_solver,
    agg,
    sa_tau: float,
    beta: float,
    device: torch.device,
) -> torch.Tensor:
    """Current-model regret on the training set, used only for optional hard mining."""
    scores = []
    model.eval()
    if agg is not None:
        agg.eval()
    with torch.no_grad():
        for batch in train_eval_loader:
            _idx, w, z, slip_compact, P_true, r_true, y_true, v_true = batch
            P_pred, r_pred, _ = model(w)
            y_star, _ = _solve_with_algo(args, P_pred, r_pred, lp_solver, qp_solver, agg=agg, sa_tau=sa_tau)
            gamma = _build_gamma(P_pred.shape[0], P_pred.shape[1], device, P_pred.dtype, start_state=_start_state())
            v_pred, _ = eval_y_on_true_mdp(P_true, r_true, y_star, gamma, beta)
            scores.append((v_true.view(-1) - v_pred).detach().cpu())
    model.train()
    if agg is not None:
        agg.train()
    return torch.cat(scores, dim=0) if scores else torch.zeros(0)


def _build_gamma(B: int, S: int, device: torch.device, dtype: torch.dtype,
                 start_state: int, start_bias: float = 0.99) -> torch.Tensor:
    # Cliff states get zero mass — they are unreachable absorbing sinks in the LP.
    n_free = S - 1 - len(CLIFF_STATES)
    base = (1.0 - start_bias) / max(1, n_free)
    g = torch.full((B, S), base, device=device, dtype=dtype)
    g[:, start_state] = start_bias
    for c in CLIFF_STATES:
        g[:, c] = 0.0
    g = g / g.sum(dim=1, keepdim=True)
    return g


def _build_cw_local_memberships(n_clusters: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Hard, row-local aggregation for Cliff Walking.

    The goal is to preserve the decision hinge near the row-2 gate while
    avoiding the fully learned dense aggregator that tends to smear the cliff
    neighborhood into the safe rows.
    """
    from mdp_cw.cliff_mdp import N_STATES

    if n_clusters == 12:
        # Gate-aware coarse partition (v2):
        # keep dedicated mass near row-2 gate while aggressively merging safe zones.
        groups = [
            [0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11],            # row 0
            [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23],# row 1
            [24, 25, 26, 27], [28, 29, 30, 31],                  # row 2 left/approach
            [32], [33, 34, 35],                                  # row 2 pre/post-gate
            [36],                                                # start
            [37, 38, 39, 40, 41, 42, 43, 44, 45, 46],           # cliff strip
            [47],                                                # goal
        ]
    elif n_clusters == 10:
        # Very coarse local partition; useful as a low-capacity ablation.
        groups = [
            [0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23],
            [24, 25, 26], [27, 28], [29, 30], [31, 32], [33, 34, 35],
            [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
        ]
    elif n_clusters == 18:
        # Medium partition (v2), almost a coarse merge of the strong M=24 layout.
        groups = [
            [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11],            # row 0
            [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23],# row 1
            [24, 25], [26, 27], [28, 29],                          # row 2 left
            [30], [31], [32], [33], [34], [35],                    # row 2 gate band
            [36],                                                   # start
            [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],          # cliff+goal tail
        ]
    elif n_clusters == 16:
        # Coarser than M=18 but still keeps the gate / cliff neighborhood local.
        groups = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
            [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23],
            [24, 25, 26], [27, 28], [29, 30], [31, 32], [33, 34, 35],
            [36], [37, 38, 39, 40, 41, 42, 43, 44, 45, 46], [47],
        ]
    elif n_clusters == 20:
        # Intermediate partition: more detail than M=16 around the row-2 gate.
        groups = [
            [0, 1], [2, 3], [4, 5], [6, 7], [8, 9, 10, 11],
            [12, 13], [14, 15], [16, 17], [18, 19], [20, 21, 22, 23],
            [24, 25], [26, 27], [28, 29], [30, 31], [32], [33], [34, 35],
            [36], [37, 38, 39, 40, 41, 42, 43, 44, 45, 46], [47],
        ]
    elif n_clusters == 24:
        # Reallocate a bit more cluster budget to the gate neighborhood
        # (row-1 cols 8..11 and row-2 cols 30..35), while compressing row 0.
        groups = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
            [12, 13], [14, 15], [16, 17], [18, 19],
            [20], [21], [22], [23],
            [24, 25], [26, 27], [28, 29], [30], [31], [32], [33], [34], [35],
            [36], [37, 38, 39, 40, 41, 42, 43, 44, 45, 46], [47],
        ]
    elif n_clusters == 28:
        groups = [
            [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11],
            [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23],
            [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35],
            [36, 37, 38], [39, 40, 41], [42, 43, 44], [45, 46, 47],
        ]
    elif n_clusters == 30:
        # Very fine partition; mostly preserves local structure at the cost of capacity.
        groups = [
            [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11],
            [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23],
            [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34], [35],
            [36], [37, 38], [39, 40], [41, 42], [43, 44], [45, 46, 47],
        ]
    else:
        raise ValueError("cw_local SSA supports only n_agg=10, 12, 16, 18, 20, 24, 28, or 30")

    W = torch.zeros(N_STATES, n_clusters, device=device, dtype=dtype)
    for m, states in enumerate(groups):
        W[states, m] = 1.0
    return W


def _build_cw_reduced_qpth_memberships(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Hand-designed coarse abstraction for CW qpth.

    Keeps the gate neighborhood local while compressing the rest of the grid
    into a small number of row/region clusters.
    """
    from mdp_cw.cliff_mdp import N_STATES, state_idx

    groups = [
        list(range(0, 12)),                       # row 0
        list(range(12, 24)),                      # row 1
        [state_idx(2, c) for c in range(0, 4)],    # row 2, left
        [state_idx(2, c) for c in range(4, 8)],    # row 2, middle-left
        [state_idx(2, 8)],                        # pre-gate
        [state_idx(2, 9)],                        # gate
        [state_idx(2, 10), state_idx(2, 11)],     # post-gate
        [state_idx(3, 0)],                        # start
        [state_idx(3, c) for c in range(1, 11)],  # cliff strip
        [state_idx(3, 11)],                       # goal
    ]
    W = torch.zeros(N_STATES, len(groups), device=device, dtype=dtype)
    for m, states in enumerate(groups):
        W[states, m] = 1.0
    return W


def eval_y_on_true_mdp(P_true: torch.Tensor, r_true: torch.Tensor,
                       y_star: torch.Tensor, gamma: torch.Tensor, beta: float):
    """Extract greedy-style policy from y_star and evaluate on the true MDP.

    Mirrors `mdp_lp.train_elasticity.eval_y_on_true_mdp` but for the
    cliff MDP (no inventory-specific shape assumptions beyond (B,S,A,S)).
    """
    B, S, A, _ = P_true.shape
    device, dtype = P_true.device, P_true.dtype
    eps = 1e-12

    if y_star.dim() == 2:
        y = y_star.view(B, S, A)
    else:
        y = y_star

    d = y.sum(dim=-1, keepdim=True)               # (B, S, 1)
    pi = y / (d + eps)                             # (B, S, A)

    P_pi = torch.einsum("bsa,bsan->bsn", pi, P_true)   # (B, S, S)
    I = torch.eye(S, device=device, dtype=dtype).unsqueeze(0)
    M = I - beta * P_pi.transpose(-1, -2)
    M = M + eps * I
    d_true = torch.linalg.solve(M, gamma.unsqueeze(-1)).squeeze(-1)   # (B, S)
    y_true_policy = pi * d_true.unsqueeze(-1)                          # (B, S, A)
    v = (y_true_policy * r_true.view(B, S, A)).sum(dim=(1, 2))         # (B,)
    return v, y_true_policy.reshape(B, S * A)


# ---------------------------------------------------------------------------
# Pretraining of the feature backbone
# ---------------------------------------------------------------------------

def pretrain_feature_net(
    feature_net: FeatureNet,
    train_loader,
    epochs: int = 60,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    verbose: bool = True,
) -> FeatureNet:
    """Supervised pretraining of FeatureNet on (w, slip_true).

    Loss: mean KL(slip_true || slip_pred) reduced to CE (slip_true is one-hot-ish).
    """
    opt = optim.Adam(feature_net.parameters(), lr=lr, weight_decay=weight_decay)
    feature_net.train()
    for epoch in range(epochs):
        total, n_batches = 0.0, 0
        for batch in train_loader:
            w, z, slip_compact, P_true, r_true, y_true, v_true = batch
            slip_pred, _ = feature_net(w)           # (B, n_slots, A, K)
            loss = _slip_ce(slip_pred, slip_compact)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(feature_net.parameters(), max_norm=10.0)
            opt.step()
            total += loss.item(); n_batches += 1
        if verbose and ((epoch + 1) % max(1, epochs // 10) == 0 or epoch == 0):
            print(f"[Pretrain] epoch {epoch+1}/{epochs}  slip_CE={total/max(1,n_batches):.4f}")
    feature_net.freeze()
    return feature_net


# ---------------------------------------------------------------------------
# Main training / evaluation
# ---------------------------------------------------------------------------

def train_eval_cliff(args, train_loader, test_loader, meta):
    """
    args fields used:
        algo: one of 'twostage', 'dfl', 'dfl_qpth'
        lr, n_epochs, grad_clip
        hidden_dim (FeatureNet hidden), feat_dim
        beta, qp_reg, loss_weight, aug_lag_rho
        pretrain_epochs, pretrain_lr
        save_dir (optional Path)
    """
    device = torch.device("cpu")
    beta = float(getattr(args, "beta", meta.get("beta", 0.95)))

    # --- infer feature dim from one batch -----------------------------------------
    w_sample, *_ = next(iter(train_loader))
    in_dim = int(w_sample.shape[-1])

    hidden = int(getattr(args, "hidden_dim", 64))
    feat_dim = int(getattr(args, "feat_dim", 32))

    # v3 plumbing: volatile cells (if any) drive the slot count on both
    # FeatureNet (pretraining target) and TransitionHead (compact slip head).
    vol_cells = list(meta.get("volatile_cells", []) or [])
    n_slots = 1 + len(vol_cells)
    if vol_cells:
        print(f"[Train] state-dependent mode: {len(vol_cells)} volatile cells "
              f"(n_slots={n_slots}) at states {vol_cells}")

    feature_net = FeatureNet(
        in_dim=in_dim, feat_dim=feat_dim, hidden=hidden, n_slots=n_slots,
    ).to(device)

    # --- Stage 1: pretrain the feature backbone -----------------------------------
    pretrain_epochs = int(getattr(args, "pretrain_epochs", 60))
    print(f"=== Experiment: cliff | algo={args.algo} | n_epochs={args.n_epochs} ===")
    if pretrain_epochs > 0:
        pretrain_feature_net(
            feature_net, train_loader,
            epochs=pretrain_epochs,
            lr=float(getattr(args, "pretrain_lr", 1e-2)),
        )
    else:
        feature_net.freeze()

    # --- Stage 2: build the decision head and optimizer ---------------------------
    head = TransitionHead(feat_dim=feat_dim, volatile_cells=vol_cells).to(device)
    model = CliffTransitionModel(feature_net, head, detach_features=True).to(device)

    # Optional soft state aggregator (only used for algo == 'dfl_ssa').
    # SoftAggDense learns memberships W[B, S, M] from the predicted (P, r) and
    # produces an aggregated MDP that the differentiable LP solves on.
    n_states_cliff = 48
    n_agg = int(getattr(args, "n_agg", 12))
    sa_tau = float(getattr(args, "sa_tau", 0.5))
    sa_r_embed = int(getattr(args, "sa_r_embed", 8))
    agg = None
    if args.algo == "dfl_ssa":
        ssa_obj = getattr(args, "ssa_objective", "cw")
        if ssa_obj == "cw_local":
            if n_agg not in (10, 12, 16, 18, 20, 24, 28, 30):
                raise ValueError("cw_local SSA expects n_agg=10, 12, 16, 18, 20, 24, 28, or 30")
            W_fixed = _build_cw_local_memberships(n_agg, device, torch.float32)
            agg = SoftAggDense(
                n_states=n_states_cliff, n_actions=N_ACTIONS,
                n_clusters=n_agg, fixed_W_matrix=W_fixed,
            ).to(device)
            optim_params = list(head.parameters())
            grad_params = list(head.parameters())
            print(f"[Train] dfl_ssa: fixed local aggregation S={n_states_cliff} -> M={n_agg}")
        elif ssa_obj == "cw_reduced_qpth":
            n_agg = 10
            W_fixed = _build_cw_reduced_qpth_memberships(device, torch.float32)
            agg = SoftAggDense(
                n_states=n_states_cliff, n_actions=N_ACTIONS,
                n_clusters=n_agg, fixed_W_matrix=W_fixed,
            ).to(device)
            optim_params = list(head.parameters())
            grad_params = list(head.parameters())
            print(f"[Train] dfl_ssa: fixed reduced-QP aggregation S={n_states_cliff} -> M={n_agg}")
        else:
            agg = SoftAggDense(
                n_states=n_states_cliff, n_actions=N_ACTIONS,
                n_clusters=n_agg, r_embed=sa_r_embed,
                learned_mode=getattr(args, "cw_agg_mode", "dest_r_bias"),
            ).to(device)
            agg_lr_ratio = float(getattr(args, "agg_lr_ratio", 0.01))
            optim_params = [
                {"params": head.parameters(), "lr": float(args.lr)},
                {"params": agg.parameters(), "lr": float(args.lr) * agg_lr_ratio},
            ]
            grad_params = list(head.parameters()) + list(agg.parameters())
            print(f"[Train] dfl_ssa: aggregating S={n_states_cliff} -> M={n_agg} "
                  f"(tau={sa_tau}, r_embed={sa_r_embed}, mode={getattr(args, 'cw_agg_mode', 'dest_r_bias')})")
    else:
        optim_params = list(head.parameters())
        grad_params = list(head.parameters())
    optimizer = optim.Adam(optim_params, lr=float(args.lr)) if args.algo != "dfl_ssa" else optim.Adam(optim_params)

    solver_obj = DenseLPSolver()
    def lp_solver(H, r, gamma):
        return solver_obj.solve(H, r, gamma)
    qp_mi = int(getattr(args, "qp_max_iter", 20))
    qp_solver = QPFunction(eps=1e-10, maxIter=qp_mi, solver=QPSolvers.PDIPM_BATCHED)
    DifferentiableLPSolver.basis_ridge = float(getattr(args, "lp_basis_ridge", 0.0))

    # Train-set mean oracle reward (reference for reward curves).
    oracle_rewards_train = []
    for batch in train_loader:
        _, _, _, _, _, _, v_true = batch
        oracle_rewards_train.append(v_true)
    train_optimal_reward = torch.cat(oracle_rewards_train).mean().item()

    # Materialize the training subset so we can build weighted samplers and
    # hard-mining replay schedules without changing the default path.
    train_tensors = _materialize_dataset(train_loader.dataset)
    train_sample_ids = torch.arange(len(train_tensors[0]), dtype=torch.long)
    train_dataset = TensorDataset(train_sample_ids, *train_tensors)
    train_batch_size = int(getattr(train_loader, "batch_size", None) or len(train_dataset) or 1)
    train_eval_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)

    # Oracle Q/hinge statistics on the row-2 cliff-adjacent states.
    hinge_mask = _cw_row2_mask(device, train_tensors[3].dtype)
    _oracle_v_state, oracle_q = _batched_oracle_q(train_tensors[3], train_tensors[4], beta)
    _oracle_best_actions, _oracle_action_gap = _state_action_gap(oracle_q)
    hinge_scores_static = _hinge_scores_from_oracle_q(oracle_q, hinge_mask)
    hinge_cutoff = float(torch.quantile(hinge_scores_static.detach().cpu(), 0.75).item()) if len(train_dataset) > 0 else 0.0

    # Bookkeeping.
    train_reward_list: list[float] = []
    train_pred_loss_list: list[float] = []
    train_dfl_loss_list: list[float] = []
    train_time_list: list[float] = []
    solver_time_sec_per_epoch: list[float] = []
    sa_time_sec_per_epoch: list[float] = []  # only populated for dfl_ssa
    backward_time_sec_per_epoch: list[float] = []
    margin_loss_list: list[float] = []
    weighted_pred_loss_list: list[float] = []
    weighted_dfl_loss_list: list[float] = []
    sample_weight_mean_list: list[float] = []
    sample_weight_max_list: list[float] = []
    train_regret_hinge_list: list[float] = []

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir: Optional[Path] = getattr(args, "save_dir", None)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Epoch-0 sanity reward (same metric as during training, before any head update).
    def _epoch_reward():
        rewards = []
        head.eval()
        if agg is not None:
            agg.eval()
        with torch.no_grad():
            for batch in train_eval_loader:
                _idx, w, z, slip_compact, P_true, r_true, y_true, v_true = batch
                P_pred, r_pred, _ = model(w)
                y_star, _ = _solve_with_algo(
                    args, P_pred, r_pred, lp_solver, qp_solver, agg=agg, sa_tau=sa_tau,
                )
                gamma = _build_gamma(P_pred.shape[0], P_pred.shape[1],
                                     device, P_pred.dtype,
                                     start_state=_start_state())
                v, _ = eval_y_on_true_mdp(P_true, r_true, y_star, gamma, beta)
                rewards.append(v.mean().item())
        head.train()
        if agg is not None:
            agg.train()
        return float(np.mean(rewards)) if rewards else 0.0

    initial_reward = _epoch_reward()
    _w = len(str(int(args.n_epochs)))
    print(f"[Train] epoch {0:>{_w}}/{args.n_epochs}  loss=N/A      "
          f"regret={train_optimal_reward - initial_reward:.3f}  oracle={train_optimal_reward:.3f}")

    # Optional static hinge weights. These are kept separate from the actual
    # objective so the default behavior (alpha=0, margin=0, mining off) is
    # exactly the old training loop.
    hinge_weight_alpha_raw = float(getattr(args, "hinge_weight_alpha", 0.0))
    hard_mine_every_raw = int(getattr(args, "hard_mine_every", 0))
    margin_lambda = float(getattr(args, "margin_lambda", 0.0))
    # Documented knobs: hinge reweight + hard mining apply only where smoothing hurts.
    _hinge_algo = args.algo in ("dfl_qpth", "dfl_ssa")
    hinge_weight_alpha = hinge_weight_alpha_raw if _hinge_algo else 0.0
    hard_mine_every = hard_mine_every_raw if _hinge_algo else 0
    if hinge_weight_alpha_raw > 0.0 and not _hinge_algo:
        print("[Train] hinge_weight_alpha ignored for algo "
              f"{args.algo!r} (use dfl_qpth or dfl_ssa).")
    if hard_mine_every_raw > 0 and not _hinge_algo:
        print("[Train] hard_mine_every ignored for algo "
              f"{args.algo!r} (use dfl_qpth or dfl_ssa).")
    margin_delta = 0.05
    hinge_gap_temp = 0.25
    static_sample_weights = 1.0 + hinge_weight_alpha * hinge_scores_static
    static_sample_weights = static_sample_weights.clamp_min(1e-6)
    replay_scores = None

    def _make_epoch_loader(weights: torch.Tensor | None = None):
        if weights is None:
            return DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        sampler = WeightedRandomSampler(
            weights.detach().cpu().double(),
            num_samples=len(train_dataset),
            replacement=True,
        )
        return DataLoader(train_dataset, batch_size=train_batch_size, sampler=sampler)

    # --- training loop ------------------------------------------------------------
    start_time = time.time()
    head.train()
    for epoch in range(int(args.n_epochs)):
        if agg is not None:
            freeze_agg = epoch < int(getattr(args, "agg_warmup_epochs", 0))
            for p_ in agg.parameters():
                p_.requires_grad = not freeze_agg

        # Optional hard-example replay uses current-model regret scores.
        if hard_mine_every > 0 and epoch > 0 and (epoch % hard_mine_every == 0):
            replay_scores = _score_train_regret(
                args, model, train_eval_loader, lp_solver, qp_solver, agg, sa_tau, beta, device
            )
        elif hard_mine_every <= 0:
            replay_scores = None

        replay_weights = None
        if replay_scores is not None and replay_scores.numel() == len(train_dataset):
            replay_weights = (1.0 + replay_scores / replay_scores.mean().clamp_min(1e-6)).clamp(0.5, 4.0)

        epoch_weights = static_sample_weights.clone()
        if replay_weights is not None:
            epoch_weights = epoch_weights * replay_weights
        use_weighted_sampler = bool(hinge_weight_alpha > 0.0 or replay_weights is not None)
        epoch_loader = _make_epoch_loader(epoch_weights if use_weighted_sampler else None)

        total_loss = 0.0
        total_pred_loss = 0.0
        total_dfl_loss = 0.0
        total_margin_loss = 0.0
        total_weighted_pred_loss = 0.0
        total_weighted_dfl_loss = 0.0
        total_sample_weight = 0.0
        total_sample_weight_max = 0.0
        epoch_solver_time = 0.0
        epoch_sa_time = 0.0
        epoch_backward_time = 0.0
        rewards_this_epoch: list[float] = []

        for batch in epoch_loader:
            sample_ids, w, z, slip_compact, P_true, r_true, y_true, v_true = batch
            sample_ids = sample_ids.long()
            B = w.shape[0]

            P_pred, r_pred, slip_pred_compact = model(w)          # forward
            gamma = _build_gamma(B, P_pred.shape[1], device, P_pred.dtype,
                                 start_state=_start_state())

            # Always compute the prediction loss (for logging / ts target).
            # CE is in compact (n_slots, A, K) space so ambient and every
            # volatile-cell override contribute equally to the gradient.
            pred_loss_ps = _slip_ce_per_sample(slip_pred_compact, slip_compact)

            if args.algo == "twostage":
                pred_loss = pred_loss_ps.mean()
                loss = pred_loss
                dfl_loss_val = 0.0
                margin_loss_val = 0.0
                # occupancy for metric only.
                with torch.no_grad():
                    y_star, solver_dt = _solve_with_algo(args, P_pred, r_pred,
                                                         lp_solver, qp_solver)
                epoch_solver_time += solver_dt

            elif args.algo in ("dfl", "dfl_qpth", "dfl_ssa", "dfl_sketch"):
                t0 = time.time()

                if args.algo == "dfl_ssa":
                    # Aggregate the predicted MDP into M clusters before solving.
                    t_agg = time.time()
                    P_solve, r_solve, gamma_solve, agg_info = agg(
                        P_pred, r_pred, gamma, tau=sa_tau,
                    )
                    epoch_sa_time += time.time() - t_agg
                    # Use generic MDP wrapper since aggregated state-space (M)
                    # differs from the cliff (S=48).
                    mdp_pred = GenericMDP(beta=beta)
                    H_pred = mdp_pred.build_H(P_solve)
                    r_flat_pred = r_solve.reshape(B, -1)
                    if ssa_obj == "cw_sketch":
                        y_agg = sketched_lp(
                            H_pred, r_flat_pred, gamma_solve,
                            n_sketches=int(getattr(args, "n_sketches", 4)),
                            sketch_ratio=float(getattr(args, "sketch_ratio", 0.1)),
                            beta=beta,
                        )
                    elif ssa_obj in ("cw_qpth", "cw_reduced_qpth"):
                        y_agg = solve_occupancy_qp_qpth(
                            H_pred, r_flat_pred, gamma_solve,
                            reg=float(getattr(args, "qp_reg", 0.1)),
                            qp_solver=qp_solver,
                        )
                    else:
                        y_agg = DifferentiableLPSolver.apply(H_pred, r_flat_pred, gamma_solve)
                    # Disaggregate back to the original (B, S*A) occupancy.
                    y_star = SoftAggDense.disaggregate_y(
                        y_agg, agg_info["W"], n_actions=N_ACTIONS,
                    )
                else:
                    mdp_pred = CliffWalkingLP(P=P_pred, r=r_pred, beta=beta)
                    H_pred = mdp_pred.build_H(P_pred)            # (B, S, SA)
                    r_flat_pred = r_pred.reshape(B, -1)

                    if args.algo == "dfl":
                        _warmup = int(getattr(args, "qpth_warmup_epochs", 0))
                        if _warmup > 0 and epoch < _warmup:
                            y_star = solve_occupancy_qp_qpth(
                                H_pred, r_flat_pred, gamma,
                                reg=float(getattr(args, "qp_reg", 0.1)),
                                qp_solver=qp_solver,
                            )
                        else:
                            _perturb_std = float(getattr(args, "dfl_perturb_std", 0.0))
                            if _perturb_std > 0.0:
                                _n_perturb = int(getattr(args, "n_perturb", 4))
                                _samples = [
                                    DifferentiableLPSolver.apply(
                                        H_pred,
                                        r_flat_pred + torch.randn_like(r_flat_pred) * _perturb_std,
                                        gamma,
                                    )
                                    for _ in range(_n_perturb)
                                ]
                                y_star = torch.stack(_samples, dim=0).mean(dim=0)
                            else:
                                y_star = DifferentiableLPSolver.apply(H_pred, r_flat_pred, gamma)
                    elif args.algo == "dfl_qpth":
                        y_star = solve_occupancy_qp_qpth(
                            H_pred, r_flat_pred, gamma,
                            reg=float(getattr(args, "qp_reg", 0.1)),
                            qp_solver=qp_solver,
                        )
                    else:  # dfl_sketch
                        y_star = sketched_lp(
                            H_pred, r_flat_pred, gamma,
                            n_sketches=int(getattr(args, "n_sketches", 4)),
                            sketch_ratio=float(getattr(args, "sketch_ratio", 0.1)),
                            beta=beta,
                        )
                epoch_solver_time += (time.time() - t0)

                # Evaluate under the TRUE MDP to form DFL loss.
                # For dfl_ssa, y_star is already disaggregated to (B, S*A).
                r_true_flat = r_true.reshape(B, -1)
                with torch.no_grad():
                    mdp_true = CliffWalkingLP(P=P_true, r=r_true, beta=beta)
                    H_true = mdp_true.build_H(P_true)
                violation = gamma - torch.einsum("bij,bj->bi", H_true, y_star)
                rho = float(getattr(args, "aug_lag_rho", 0.1))
                aug_lag_ps = rho * violation.pow(2).mean(dim=1)
                v_true_pred_ps = torch.sum(r_true_flat * y_star, dim=1)
                dfl_loss_ps = -v_true_pred_ps + aug_lag_ps

                # Margin / ranking auxiliary loss:
                #   q_pred(s,a) = r_pred(s,a) + beta * E[V*_true(s')]
                #   L_margin = relu(delta - (q_pred(s,a*) - max_{a!=a*} q_pred(s,a)))
                margin_loss_ps = torch.zeros_like(pred_loss_ps)
                if margin_lambda > 0.0 and args.algo in ("dfl", "dfl_qpth", "dfl_ssa", "dfl_sketch"):
                    v_true_state, q_true = _batched_oracle_q(P_true, r_true, beta)
                    oracle_best = q_true.argmax(dim=-1)
                    state_importance = torch.exp(
                        -_state_action_gap(q_true)[1] / max(hinge_gap_temp, 1e-6)
                    ) * hinge_mask.view(1, -1)
                    q_pred_oracle = r_pred + beta * torch.einsum("bsan,bn->bsa", P_pred, v_true_state)
                    q_star = q_pred_oracle.gather(2, oracle_best.unsqueeze(-1)).squeeze(-1)
                    other = q_pred_oracle.masked_fill(
                        F.one_hot(oracle_best, num_classes=q_pred_oracle.shape[-1]).bool(),
                        float("-inf"),
                    ).max(dim=-1).values
                    margin_gap = q_star - other
                    margin_violation = F.relu(margin_delta - margin_gap)
                    margin_loss_ps = (
                        state_importance * margin_violation
                    ).sum(dim=1) / state_importance.sum(dim=1).clamp_min(1.0)

                sample_weights = static_sample_weights[sample_ids]
                if replay_weights is not None:
                    sample_weights = sample_weights * replay_weights[sample_ids]
                sample_weights = sample_weights.to(device=device, dtype=pred_loss_ps.dtype)
                loss_terms = (
                    dfl_loss_ps
                    + float(getattr(args, "loss_weight", 1.0)) * pred_loss_ps
                    + margin_lambda * margin_loss_ps
                )
                loss = (sample_weights * loss_terms).sum() / sample_weights.sum().clamp_min(1e-6)
                dfl_loss_val = float(dfl_loss_ps.mean().item())
                margin_loss_val = float(margin_loss_ps.mean().item())
                pred_loss = pred_loss_ps.mean()
                total_weighted_pred_loss += float((sample_weights * pred_loss_ps).sum().item())
                total_weighted_dfl_loss += float((sample_weights * dfl_loss_ps).sum().item())
                total_margin_loss += float((sample_weights * margin_loss_ps).sum().item())
                total_sample_weight += float(sample_weights.sum().item())
                total_sample_weight_max = max(total_sample_weight_max, float(sample_weights.max().item()))
            else:
                raise ValueError(f"Unknown algo: {args.algo}")

            optimizer.zero_grad()
            _t_bwd = time.time()
            loss.backward()
            epoch_backward_time += time.time() - _t_bwd
            torch.nn.utils.clip_grad_norm_(grad_params,
                                           max_norm=float(getattr(args, "grad_clip", 10.0)))
            optimizer.step()

            total_loss += float(loss.item())
            total_pred_loss += float(pred_loss.item())
            total_dfl_loss += dfl_loss_val

            with torch.no_grad():
                y_eval = y_star.detach()
                if args.algo == "dfl_sketch":
                    mdp_eval = CliffWalkingLP(P=P_pred.detach(), r=r_pred.detach(), beta=beta)
                    y_eval, _, _, _ = solve_lp(mdp_eval, lp_solver)
                elif args.algo == "dfl_ssa" and getattr(args, "ssa_objective", "cw") == "cw_sketch":
                    y_eval, _ = _solve_with_algo(
                        args, P_pred.detach(), r_pred.detach(),
                        lp_solver, qp_solver, agg=agg, sa_tau=sa_tau,
                    )
                elif args.algo == "dfl_ssa" and getattr(args, "ssa_objective", "cw") in ("cw_qpth", "cw_reduced_qpth"):
                    y_eval, _ = _solve_with_algo(
                        args, P_pred.detach(), r_pred.detach(),
                        lp_solver, qp_solver, agg=agg, sa_tau=sa_tau,
                    )
                v, _ = eval_y_on_true_mdp(P_true, r_true, y_eval, gamma, beta)
                rewards_this_epoch.append(v.mean().item())

        batch_denom = max(1, len(epoch_loader))
        train_reward_list.append(float(np.mean(rewards_this_epoch)))
        train_pred_loss_list.append(total_pred_loss / batch_denom)
        train_dfl_loss_list.append(total_dfl_loss / batch_denom)
        if total_sample_weight > 0.0:
            weighted_pred = total_weighted_pred_loss / total_sample_weight
            weighted_dfl = total_weighted_dfl_loss / total_sample_weight
            weighted_margin = total_margin_loss / total_sample_weight
            mean_sample_weight = total_sample_weight / max(1, len(train_dataset))
        else:
            weighted_pred = total_pred_loss / batch_denom
            weighted_dfl = total_dfl_loss / batch_denom
            weighted_margin = 0.0
            mean_sample_weight = 1.0
        weighted_pred_loss_list.append(float(weighted_pred))
        weighted_dfl_loss_list.append(float(weighted_dfl))
        margin_loss_list.append(float(weighted_margin))
        sample_weight_mean_list.append(float(mean_sample_weight))
        sample_weight_max_list.append(float(total_sample_weight_max))

        train_regret_hinge_list.append(float(weighted_margin))

        train_time_list.append(time.time() - start_time)
        solver_time_sec_per_epoch.append(epoch_solver_time)
        sa_time_sec_per_epoch.append(epoch_sa_time)
        backward_time_sec_per_epoch.append(epoch_backward_time)

        _epoch_regret = train_optimal_reward - train_reward_list[-1]
        msg = (
            f"[Train] epoch {epoch+1:>{_w}}/{args.n_epochs}  "
            f"loss={total_loss/max(1,batch_denom):.4f}  "
            f"regret={_epoch_regret:.3f}  oracle={train_optimal_reward:.3f}"
        )
        if hinge_weight_alpha > 0.0 or margin_lambda > 0.0 or hard_mine_every > 0:
            msg += (
                f"  wCE={weighted_pred_loss_list[-1]:.4f}"
                f"  wDFL={weighted_dfl_loss_list[-1]:.4f}"
                f"  margin={margin_loss_list[-1]:.4f}"
            )
        print(msg)

        # lightweight checkpoint
        if save_dir is not None:
            ckpt = {
                "epoch": epoch + 1,
                "train_reward_per_epoch": [initial_reward] + train_reward_list,
                "train_pred_loss_per_epoch": train_pred_loss_list,
                "train_dfl_loss_per_epoch": train_dfl_loss_list,
                "train_weighted_pred_loss_per_epoch": weighted_pred_loss_list,
                "train_weighted_dfl_loss_per_epoch": weighted_dfl_loss_list,
                "train_margin_loss_per_epoch": margin_loss_list,
                "train_sample_weight_mean_per_epoch": sample_weight_mean_list,
                "train_sample_weight_max_per_epoch": sample_weight_max_list,
                "train_regret_hinge_per_epoch": train_regret_hinge_list,
                "train_time": train_time_list,
                "solver_time_sec_per_epoch": solver_time_sec_per_epoch,
                "backward_time_sec_per_epoch": backward_time_sec_per_epoch,
                "algo": args.algo,
            }
            (save_dir / "checkpoint.json").write_text(json.dumps(ckpt, indent=2))
            torch.save(head.state_dict(), save_dir / "head_state.pt")
            torch.save(feature_net.state_dict(), save_dir / "feature_net_state.pt")

    print(f"[Train] done  elapsed={time.time()-start_time:.1f}s")
    head.eval()
    if agg is not None:
        agg.eval()
    test_obj_values: list[float] = []
    test_oracle_values: list[float] = []
    test_slip_ce: list[float] = []
    test_slip_pred_all, test_slip_true_all = [], []
    test_regime_ids: list[int] = []
    test_ambient_slip_chunks: list[np.ndarray] = []
    test_hinge_scores: list[float] = []
    with torch.no_grad():
        for batch in test_loader:
            w, z, slip_compact, P_true, r_true, y_true, v_true = batch
            B = w.shape[0]
            P_pred, r_pred, slip_pred_compact = model(w)
            gamma = _build_gamma(B, P_pred.shape[1], device, P_pred.dtype,
                                 start_state=_start_state())
            # Predicted: solve on predicted MDP, evaluate on true MDP.
            y_star_pred, _ = _solve_with_algo(args, P_pred, r_pred, lp_solver, qp_solver,
                                              agg=agg, sa_tau=sa_tau)
            v_pred, _ = eval_y_on_true_mdp(P_true, r_true, y_star_pred, gamma, beta)
            test_obj_values.extend(v_pred.tolist())
            test_oracle_values.extend(v_true.tolist())
            test_slip_ce.append(_slip_ce(slip_pred_compact, slip_compact).item())
            test_slip_pred_all.append(slip_pred_compact)
            test_slip_true_all.append(slip_compact)
            test_regime_ids.extend(z.long().view(-1).tolist())
            test_ambient_slip_chunks.append(
                slip_pred_compact[:, 0, ...].detach().cpu().numpy()
            )
            _, q_true_test = _batched_oracle_q(P_true, r_true, beta)
            test_hinge_scores.extend(_hinge_scores_from_oracle_q(q_true_test, hinge_mask).cpu().tolist())

    amb_slip = (
        np.concatenate(test_ambient_slip_chunks, axis=0)
        if test_ambient_slip_chunks
        else np.zeros((0, N_ACTIONS, N_SLIP), dtype=np.float64)
    )
    z_test = np.asarray(test_regime_ids, dtype=np.int64)
    rnames = _regime_display_names(meta)
    by_reg: dict[str, float] = {}
    reg_counts: dict[str, int] = {}
    test_obj_values_t = torch.tensor(test_obj_values)
    test_oracle_values_t = torch.tensor(test_oracle_values)
    test_regret = test_oracle_values_t - test_obj_values_t
    r_ = test_regret.cpu().numpy()
    hinge_scores_np = np.asarray(test_hinge_scores, dtype=np.float64)
    if hinge_scores_np.size > 0:
        hinge_cut = float(np.quantile(hinge_scores_np, 0.75))
        hinge_subset = hinge_scores_np >= hinge_cut
        test_regret_hinge = float(r_[hinge_subset].mean()) if np.any(hinge_subset) else float("nan")
    else:
        hinge_cut = float("nan")
        hinge_subset = np.zeros_like(r_, dtype=bool)
        test_regret_hinge = float("nan")
    for i, name in enumerate(rnames):
        m = z_test == i
        reg_counts[name] = int(m.sum())
        by_reg[name] = float(r_[m].mean()) if np.any(m) else float("nan")

    result = {
        "train_reward_per_epoch": [initial_reward] + train_reward_list,
        "train_pred_loss_per_epoch": train_pred_loss_list,
        "train_dfl_loss_per_epoch": train_dfl_loss_list,
        "train_weighted_pred_loss_per_epoch": weighted_pred_loss_list,
        "train_weighted_dfl_loss_per_epoch": weighted_dfl_loss_list,
        "train_margin_loss_per_epoch": margin_loss_list,
        "train_sample_weight_mean_per_epoch": sample_weight_mean_list,
        "train_sample_weight_max_per_epoch": sample_weight_max_list,
        "train_regret_hinge_per_epoch": train_regret_hinge_list,
        "train_time": train_time_list,
        "solver_time_sec_per_epoch": solver_time_sec_per_epoch,
        "sa_time_sec_per_epoch": sa_time_sec_per_epoch,
        "backward_time_sec_per_epoch": backward_time_sec_per_epoch,
        "test_obj_values": test_obj_values_t.cpu().numpy(),
        "test_oracle_values": test_oracle_values_t.cpu().numpy(),
        "test_regret": test_regret.cpu().numpy(),
        "test_regime_ids": test_regime_ids,
        "test_mean_regret_by_regime": by_reg,
        "test_regime_counts": reg_counts,
        "test_ambient_slip_pred": amb_slip.tolist(),
        "test_slip_ce": float(np.mean(test_slip_ce)) if test_slip_ce else 0.0,
        "test_mean_pred_reward": float(test_obj_values_t.mean().item()),
        "test_mean_oracle_reward": float(test_oracle_values_t.mean().item()),
        "test_mean_regret": float(test_regret.mean().item()),
        "test_mean_regret_hinge": test_regret_hinge,
        "test_hinge_score_cutoff": hinge_cut,
        "test_hinge_subset_count": int(np.sum(hinge_subset)),
        "test_hinge_score_mean": float(hinge_scores_np.mean()) if hinge_scores_np.size > 0 else float("nan"),
        "train_optimal_reward": train_optimal_reward,
        "algo": args.algo,
    }

    print(f"[Eval]    oracle={result['test_mean_oracle_reward']:.3f}  "
          f"pred={result['test_mean_pred_reward']:.3f}  "
          f"regret={result['test_mean_regret']:.3f}  "
          f"slip_ce={result['test_slip_ce']:.4f}")

    if save_dir is not None:
        # Save a final copy.
        serializable = {k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in result.items()}
        (save_dir / "results.json").write_text(json.dumps(serializable, indent=2))

        # Auto-generate per-run plots.
        try:
            from mdp_cw.plot_cliff_run import make_all_plots
            make_all_plots(save_dir)
        except Exception as e:
            print(f"[Train] WARN: per-run plotting failed ({e})")

    return result


# ---------------------------------------------------------------------------
# LP/QP solver dispatch
# ---------------------------------------------------------------------------
def _start_state() -> int:
    # Kept local to break the circular reference between args and cliff_mdp.
    from mdp_cw.cliff_mdp import START_STATE
    return START_STATE


def _solve_with_algo(args, P_pred: torch.Tensor, r_pred: torch.Tensor,
                     lp_solver, qp_solver, agg=None, sa_tau: float = 0.5):
    """Solve the LP/QP on the PREDICTED MDP.  Returns (y_star, solver_dt).

    For algo == 'dfl_ssa', `agg` must be a SoftAggDense module: we aggregate
    the predicted MDP into M clusters, solve the differentiable LP on the
    aggregated MDP, and disaggregate y_star back to (B, S*A).
    """
    beta = float(getattr(args, "beta", 0.95))
    B = P_pred.shape[0]
    S = P_pred.shape[1]
    t0 = time.time()
    if args.algo in ("twostage",):
        # Use Gurobi; gradients not needed for metric-only solve.
        mdp = CliffWalkingLP(P=P_pred, r=r_pred, beta=beta)
        y_star, _, _, _ = solve_lp(mdp, lp_solver)
    elif args.algo == "dfl":
        mdp = CliffWalkingLP(P=P_pred, r=r_pred, beta=beta)
        H = mdp.build_H(P_pred)
        r_flat = r_pred.reshape(B, -1)
        gamma = mdp.build_gamma(B, S, P_pred.device)
        y_star = DifferentiableLPSolver.apply(H, r_flat, gamma)
    elif args.algo == "dfl_qpth":
        mdp = CliffWalkingLP(P=P_pred, r=r_pred, beta=beta)
        H = mdp.build_H(P_pred)
        r_flat = r_pred.reshape(B, -1)
        gamma = mdp.build_gamma(B, S, P_pred.device)
        y_star = solve_occupancy_qp_qpth(
            H, r_flat, gamma,
            reg=float(getattr(args, "qp_reg", 0.1)),
            qp_solver=qp_solver,
        )
    elif args.algo == "dfl_sketch":
        mdp = CliffWalkingLP(P=P_pred, r=r_pred, beta=beta)
        y_star, _, _, _ = solve_lp(mdp, lp_solver)
    elif args.algo == "dfl_ssa":
        if agg is None:
            raise RuntimeError("dfl_ssa requires an aggregator")
        _mdp_ssa = CliffWalkingLP(P=P_pred, r=r_pred, beta=beta)
        gamma = _mdp_ssa.gamma
        P_solve, r_solve, gamma_solve, info = agg(
            P_pred, r_pred, gamma, tau=sa_tau,
        )
        mdp = GenericMDP(beta=beta)
        H = mdp.build_H(P_solve)
        r_flat = r_solve.reshape(B, -1)
        # Evaluation/metric solves should match the selected aggregate objective.
        ssa_obj = getattr(args, "ssa_objective", "cw")
        if ssa_obj in ("cw_qpth", "cw_reduced_qpth"):
            y_agg = solve_occupancy_qp_qpth(
                H, r_flat, gamma_solve,
                reg=float(getattr(args, "qp_reg", 0.1)),
                qp_solver=qp_solver,
            )
        elif ssa_obj == "cw_sketch":
            y_agg = sketched_lp(
                H, r_flat, gamma_solve,
                n_sketches=int(getattr(args, "n_sketches", 4)),
                sketch_ratio=float(getattr(args, "sketch_ratio", 0.1)),
                beta=beta,
            )
        elif ssa_obj == "cw_local":
            y_agg = DifferentiableLPSolver.apply(H, r_flat, gamma_solve)
        else:
            y_agg = DifferentiableLPSolver.apply(H, r_flat, gamma_solve)
        y_star = SoftAggDense.disaggregate_y(
            y_agg, info["W"], n_actions=N_ACTIONS,
        )
    else:
        raise ValueError(f"Unknown algo {args.algo}")
    return y_star, (time.time() - t0)


__all__ = ["pretrain_feature_net", "train_eval_cliff", "eval_y_on_true_mdp"]
