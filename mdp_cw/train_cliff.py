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

import copy
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
from qpth.qp import QPFunction, QPSolvers

from mdp_cw.cliff_mdp import CLIFF_STATES, CliffWalkingLP, N_ACTIONS, N_SLIP
from mdp_cw.features_model import CliffTransitionModel, FeatureNet, TransitionHead
from mdp_cw.weather_dynamics import (
    REGIME_NAMES_V1,
    REGIME_NAMES_V2,
    REGIME_NAMES_V3,
    REGIME_NAMES_V4,
)
from mdp_lp.mdp import MDP as GenericMDP
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver, solve_lp_gurobi
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

    # Optional validation split carved out of the TRAINING set. Used only to
    # pick the epoch to evaluate (early stopping); the test set is never seen.
    # Selecting on train reward instead would favour epochs that memorise which
    # individual training samples are hazardous.
    val_loader = None
    _vf = float(getattr(args, "early_stop_val_frac", 0.0) or 0.0)
    if _vf > 0.0:
        from torch.utils.data import DataLoader, Subset
        _ds = train_loader.dataset
        _n = len(_ds)
        _nv = max(1, int(round(_n * _vf)))
        _g = torch.Generator().manual_seed(int(getattr(args, "seed", 0)) + 12345)
        _perm = torch.randperm(_n, generator=_g).tolist()
        _bs = int(getattr(args, "batch_size", 32))
        val_loader = DataLoader(Subset(_ds, _perm[:_nv]), batch_size=_bs, shuffle=False)
        train_loader = DataLoader(Subset(_ds, _perm[_nv:]), batch_size=_bs, shuffle=True)
        print(f"[Train] early-stop validation split: {_nv} val / {_n - _nv} train")

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
    if pretrain_epochs > 0:
        print(f"[Train] Pretraining FeatureNet for {pretrain_epochs} epochs...")
        pretrain_feature_net(
            feature_net, train_loader,
            epochs=pretrain_epochs,
            lr=float(getattr(args, "pretrain_lr", 1e-2)),
        )
    else:
        feature_net.freeze()

    # --- Stage 2: build the decision head and optimizer ---------------------------
    head = TransitionHead(
        feat_dim=feat_dim, volatile_cells=vol_cells,
        ambient_only=bool(getattr(args, "misspec_ambient", False)),
    ).to(device)
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
        agg = SoftAggDense(
            n_states=n_states_cliff, n_actions=N_ACTIONS,
            n_clusters=n_agg, r_embed=sa_r_embed,
        ).to(device)
        params = list(head.parameters()) + list(agg.parameters())
        print(f"[Train] dfl_ssa: aggregating S={n_states_cliff} -> M={n_agg} "
              f"(tau={sa_tau}, r_embed={sa_r_embed})")
    else:
        params = list(head.parameters())
    optimizer = optim.Adam(params, lr=float(args.lr))

    solver_obj = DenseLPSolver()
    def lp_solver(H, r, gamma):
        return solver_obj.solve(H, r, gamma)
    qp_mi = int(getattr(args, "qp_max_iter", 20))
    qp_solver = QPFunction(eps=1e-10, maxIter=qp_mi, solver=QPSolvers.PDIPM_BATCHED)

    # Train-set mean oracle reward (reference for reward curves).
    oracle_rewards_train = []
    for batch in train_loader:
        _, _, _, _, _, _, v_true = batch
        oracle_rewards_train.append(v_true)
    train_optimal_reward = torch.cat(oracle_rewards_train).mean().item()

    # Bookkeeping.
    train_reward_list: list[float] = []
    train_pred_loss_list: list[float] = []
    train_dfl_loss_list: list[float] = []
    train_time_list: list[float] = []
    solver_time_sec_per_epoch: list[float] = []
    sa_time_sec_per_epoch: list[float] = []  # only populated for dfl_ssa
    backward_time_sec_per_epoch: list[float] = []

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
            for batch in train_loader:
                w, z, slip_compact, P_true, r_true, y_true, v_true = batch
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
    print(f"[Train] initial reward={initial_reward:.3f}  "
          f"oracle mean={train_optimal_reward:.3f}")

    # --- training loop ------------------------------------------------------------
    start_time = time.time()
    # Early stopping on TRAIN decision quality (uses training labels only, exactly
    # like the two-stage CE objective does). Guards against DFL runs that wander
    # into over-optimistic predictors late in training.
    _es_on = bool(getattr(args, "early_stop_on_train_reward", False))
    _es_best = float("-inf")
    _es_head = None
    _es_agg = None
    head.train()
    for epoch in range(int(args.n_epochs)):
        total_loss = 0.0
        total_pred_loss = 0.0
        total_dfl_loss = 0.0
        epoch_solver_time = 0.0
        epoch_sa_time = 0.0
        epoch_backward_time = 0.0
        rewards_this_epoch: list[float] = []

        for batch_idx, batch in enumerate(train_loader):
            w, z, slip_compact, P_true, r_true, y_true, v_true = batch
            B = w.shape[0]

            P_pred, r_pred, slip_pred_compact = model(w)          # forward
            gamma = _build_gamma(B, P_pred.shape[1], device, P_pred.dtype,
                                 start_state=_start_state())

            # Always compute the prediction loss (for logging / ts target).
            # CE is in compact (n_slots, A, K) space so ambient and every
            # volatile-cell override contribute equally to the gradient.
            pred_loss = _slip_ce(slip_pred_compact, slip_compact)

            if args.algo == "twostage":
                loss = pred_loss
                dfl_loss_val = 0.0
                # occupancy for metric only.
                with torch.no_grad():
                    y_star, solver_dt = _solve_with_algo(args, P_pred, r_pred,
                                                         lp_solver, qp_solver)
                epoch_solver_time += solver_dt

            elif args.algo in ("dfl", "dfl_qpth", "dfl_ssa", "dfl_sketch", "dfl_feas") and \
                    epoch < int(getattr(args, "head_ce_warmup_epochs", 0)):
                # CE-only warmup for the freshly initialized TransitionHead:
                # unlike mdp_lp, the head is NOT pretrained, so with a small
                # loss_weight the DFL phase would otherwise start from random
                # predictions. Train pure CE first; the DFL loss takes over
                # afterwards. Solve is metric-only here.
                loss = pred_loss
                dfl_loss_val = 0.0
                with torch.no_grad():
                    y_star, solver_dt = _solve_with_algo(
                        args, P_pred, r_pred, lp_solver, qp_solver,
                        agg=agg, sa_tau=sa_tau,
                    )
                epoch_solver_time += solver_dt
            elif args.algo in ("dfl", "dfl_qpth", "dfl_ssa", "dfl_sketch", "dfl_feas"):
                t0 = time.time()

                agg_info = None
                if args.algo == "dfl_feas":
                    # DFL-Feas: Odece (Mandi et al., NeurIPS'25) adapted to the
                    # occupancy LP, ported from mdp_lp/train_elasticity.py.
                    # Solver-free gradients: y^- is solved under no_grad on the
                    # PREDICTED MDP and y_true comes from the dataset, so both
                    # are constants. The predictor is reached only through
                    # c(y; th) = H_pred y - gamma.
                    mdp_pred = CliffWalkingLP(P=P_pred, r=r_pred, beta=beta)
                    H_pred = mdp_pred.build_H(P_pred)          # differentiable
                    r_flat_pred = r_pred.reshape(B, -1)
                    with torch.no_grad():
                        y_neg, _, _, _ = solve_lp(
                            CliffWalkingLP(P=P_pred.detach(), r=r_pred.detach(),
                                           beta=beta), lp_solver)
                    y_star = y_neg
                elif args.algo == "dfl_ssa":
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
                    # Sketched LP smooths the basis-flip gradient that makes
                    # plain DifferentiableLPSolver.apply unstable on top of a
                    # learnable aggregator. Mirrors mdp_lp's dfl_sketch+n_agg.
                    y_agg = sketched_lp(
                        H_pred, r_flat_pred, gamma_solve,
                        n_sketches=int(getattr(args, "n_sketches", 4)),
                        sketch_ratio=float(getattr(args, "sketch_ratio", 0.7)),
                        sketch_mode=str(getattr(args, "sketch_mode", "gaussian")),
                        beta=beta,
                    )
                    # Disaggregate back to the original (B, S*A) occupancy for
                    # downstream policy evaluation on the full MDP.
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
                            _pstd = float(getattr(args, "dfl_perturb_std", 0.0) or 0.0)
                            if _pstd > 0.0:
                                # PFYL-style perturbation: average the LP solution
                                # over K reward draws so the gradient averages over
                                # several optimal bases instead of the single basis
                                # at the current prediction.
                                _K = int(getattr(args, "dfl_n_perturb", 4))
                                _ys = [
                                    DifferentiableLPSolver.apply(
                                        H_pred,
                                        r_flat_pred + _pstd * torch.randn_like(r_flat_pred),
                                        gamma,
                                    )
                                    for _ in range(_K)
                                ]
                                y_star = torch.stack(_ys, dim=0).mean(dim=0)
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

                rho = float(getattr(args, "aug_lag_rho", 0.1))
                # loss_mode in {regret, penalty, lagrangian, auglag}; default
                # 'penalty' reproduces the original Eq.(5) loss EXACTLY.
                loss_mode = str(getattr(args, "loss_mode", "penalty"))
                if args.algo == "dfl_ssa" and agg_info is not None:
                    # Aug-lag in the AGGREGATED true MDP (mirrors mdp_lp).
                    # Aggregating the true MDP with the same W keeps the
                    # constraint LHS in M-dim space where y_agg lives, which is
                    # numerically much better behaved than projecting back to
                    # S=48 and computing violation against the full H_true.
                    W = agg_info["W"]                                       # (B, S, M)
                    mass = W.sum(dim=1).clamp_min(1e-8)                     # (B, M)
                    r_true_agg = torch.einsum(
                        "bsm,bsa->bma", W, r_true
                    ) / mass.unsqueeze(-1)                                  # (B, M, A)
                    right_t = torch.einsum("bsaj,bjm->bsam", P_true, W)     # (B, S, A, M)
                    p_hat_t = torch.einsum("bsm,bsan->bman", W, right_t)    # (B, M, A, M)
                    P_true_agg = p_hat_t / mass[:, :, None, None]
                    P_true_agg = P_true_agg.clamp_min(0.0)
                    P_true_agg = P_true_agg / P_true_agg.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-8)
                    gamma_true_agg = torch.einsum("bs,bsm->bm", gamma, W).clamp_min(1e-8)
                    gamma_true_agg = gamma_true_agg / gamma_true_agg.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1e-8)
                    mdp_true_agg = GenericMDP(beta=beta)
                    H_true_agg = mdp_true_agg.build_H(P_true_agg)           # (B, M, M*A)
                    r_true_agg_flat = r_true_agg.reshape(B, -1)
                    violation = gamma_true_agg - torch.einsum(
                        "bij,bj->bi", H_true_agg, y_agg
                    )
                    aug_lag = rho * violation.pow(2).mean()
                    v_true_pred = torch.sum(r_true_agg_flat * y_agg, dim=1).mean()
                    if loss_mode in ("lagrangian", "auglag"):
                        # Linear dual term needs the dual of the AGGREGATED true
                        # LP. H̄, γ̄ change with W every step, so it is solved
                        # fresh under no_grad — a constant wrt θ (mirrors
                        # mdp_lp/train_elasticity).
                        with torch.no_grad():
                            y_agg_true, _, dual_agg = solve_lp_gurobi(
                                H_true_agg.detach(), r_true_agg_flat.detach(),
                                gamma_true_agg.detach())
                        if batch_idx == 0:
                            # Strong-duality gate: γ̄ᵀν̄ == r̄ᵀȳ* at the
                            # aggregated-true optimum; a flipped dual sign would
                            # double the bias instead of cancelling it.
                            lhs = (gamma_true_agg.detach() * dual_agg).sum(dim=1)
                            rhs = (r_true_agg_flat.detach() * y_agg_true).sum(dim=1)
                            if not torch.allclose(lhs, rhs, rtol=1e-2, atol=1e-2):
                                rel = ((lhs - rhs).abs() / (rhs.abs() + 1e-8)).max().item()
                                raise RuntimeError(
                                    "Aggregated linear-term consistency check failed: "
                                    f"gamma_agg^T dual != r_agg^T y*_agg (max rel err {rel:.3e}); "
                                    "dual sign convention inconsistent — auglag results untrustworthy.")
                        linear_term = -(dual_agg * violation).sum(dim=1).mean()
                elif args.algo == "dfl_feas":
                    # Each equality is the softplus pair of its two inequality
                    # sides; UNSAT sides come from the sign of the TRUE violation
                    # of y^- (tolerance odece_tol, since exact equality
                    # feasibility is measure-zero).
                    ups = float(getattr(args, "odece_margin", 0.01))
                    tol = float(getattr(args, "odece_tol", 1e-4))
                    o_alpha = float(getattr(args, "odece_alpha", 0.5))
                    mse_w = float(getattr(args, "odece_mse_weight", 0.0))
                    r_true_flat = r_true.reshape(B, -1)
                    with torch.no_grad():
                        mdp_true = CliffWalkingLP(P=P_true, r=r_true, beta=beta)
                        H_true = mdp_true.build_H(P_true)
                    violation = gamma - torch.einsum("bij,bj->bi", H_true, y_star)
                    v_true_pred = torch.sum(r_true_flat * y_star, dim=1).mean()
                    # OPL: the true optimum should satisfy the PREDICTED flow
                    # constraints.
                    c_true = torch.einsum("bij,bj->bi", H_pred, y_true) - gamma
                    opl = (F.softplus(ups + c_true)
                           + F.softplus(ups - c_true)).sum(dim=1).mean()
                    # IPL: at c_neg = 0 the value is constant but the gradient
                    # -sigma(ups)*dc/dtheta pushes the predicted residual at y^-
                    # toward the sign of its true violation.
                    c_neg = torch.einsum("bij,bj->bi", H_pred, y_star) - gamma
                    v_neg = -violation                     # = H_true y^- - gamma
                    unsat_pos = (v_neg > tol).float()
                    unsat_neg = (v_neg < -tol).float()
                    infeas = (v_neg.abs().amax(dim=1, keepdim=True) > tol).float()
                    ipl = (infeas * (unsat_pos * F.softplus(ups - c_neg)
                                     + unsat_neg * F.softplus(ups + c_neg))).sum(dim=1).mean()
                    dfl_loss = o_alpha * ipl + (1.0 - o_alpha) * opl
                    if mse_w > 0:
                        dfl_loss = dfl_loss + mse_w * F.mse_loss(r_flat_pred, r_true_flat)
                    loss_mode = None       # skip the shared loss_mode branches
                else:
                    r_true_flat = r_true.reshape(B, -1)
                    with torch.no_grad():
                        mdp_true = CliffWalkingLP(P=P_true, r=r_true, beta=beta)
                        H_true = mdp_true.build_H(P_true)
                    violation = gamma - torch.einsum("bij,bj->bi", H_true, y_star)
                    aug_lag = rho * violation.pow(2).mean()
                    v_true_pred = torch.sum(r_true_flat * y_star, dim=1).mean()
                    if loss_mode in ("lagrangian", "auglag"):
                        # Linear dual term (v*)^T(H_true y - gamma) = -(v*)^T violation.
                        # The true MDP is fixed per sample; solving under no_grad
                        # keeps the dual a constant wrt θ.
                        with torch.no_grad():
                            y_true_lp, _, dual_true = solve_lp_gurobi(
                                H_true, r_true_flat.detach(), gamma)
                        if batch_idx == 0:
                            lhs = (gamma * dual_true).sum(dim=1)
                            rhs = (r_true_flat.detach() * y_true_lp).sum(dim=1)
                            if not torch.allclose(lhs, rhs, rtol=1e-2, atol=1e-2):
                                rel = ((lhs - rhs).abs() / (rhs.abs() + 1e-8)).max().item()
                                raise RuntimeError(
                                    "Linear-term consistency check failed: "
                                    f"gamma^T dual != r^T y*_true (max rel err {rel:.3e}); "
                                    "dual sign convention inconsistent — auglag results untrustworthy.")
                        linear_term = -(dual_true * violation).sum(dim=1).mean()
                if loss_mode is not None:      # dfl_feas builds its own loss
                    dfl_loss = -v_true_pred
                if loss_mode in ("penalty", "auglag"):
                    dfl_loss = dfl_loss + aug_lag
                if loss_mode in ("lagrangian", "auglag"):
                    dfl_loss = dfl_loss + linear_term
                aw = getattr(args, "ambient_ce_weight", None)
                if aw is not None and slip_pred_compact.dim() == 4 and slip_pred_compact.shape[1] > 1:
                    # Split CE anchoring: clamp the ambient slot (shared by all
                    # non-volatile cells; corrupting it wrecks the whole MDP)
                    # while loss_weight applies only to the volatile slots the
                    # decision gradient needs to move.
                    ce_amb = _slip_ce(slip_pred_compact[:, :1], slip_compact[:, :1])
                    ce_vol = _slip_ce(slip_pred_compact[:, 1:], slip_compact[:, 1:])
                    loss = dfl_loss + float(aw) * ce_amb \
                        + float(getattr(args, "loss_weight", 1.0)) * ce_vol
                else:
                    loss = dfl_loss + float(getattr(args, "loss_weight", 1.0)) * pred_loss
                dfl_loss_val = float(dfl_loss.item())
            else:
                raise ValueError(f"Unknown algo: {args.algo}")

            optimizer.zero_grad()
            _t_bwd = time.time()
            loss.backward()
            epoch_backward_time += time.time() - _t_bwd
            torch.nn.utils.clip_grad_norm_(params,
                                           max_norm=float(getattr(args, "grad_clip", 10.0)))
            optimizer.step()

            total_loss += float(loss.item())
            total_pred_loss += float(pred_loss.item())
            total_dfl_loss += dfl_loss_val

            with torch.no_grad():
                y_eval = y_star.detach()
                # For sketch- and aggregator-smoothed paths the y returned from
                # the surrogate is averaged / disaggregated and does NOT
                # represent a deployable policy on the original MDP. Solve the
                # plain LP on the (detached) predicted MDP for the metric,
                # mirroring how mdp_lp evaluates dfl_sketch / dfl_ssa.
                # During the CE-warmup epochs y_star ALREADY is that plain-LP
                # solution (_solve_with_algo takes the same branch), so solving
                # again would be a bit-identical duplicate: ~40% of the epochs
                # here paid for two identical Gurobi solves per batch.
                if args.algo in ("dfl_qpth", "dfl_sketch", "dfl_ssa") and \
                        epoch >= int(getattr(args, "head_ce_warmup_epochs", 0)):
                    mdp_eval = CliffWalkingLP(P=P_pred.detach(), r=r_pred.detach(), beta=beta)
                    y_eval, _, _, _ = solve_lp(mdp_eval, lp_solver)
                v, _ = eval_y_on_true_mdp(P_true, r_true, y_eval, gamma, beta)
                rewards_this_epoch.append(v.mean().item())

        train_reward_list.append(float(np.mean(rewards_this_epoch)))
        _es_metric = train_reward_list[-1]
        if _es_on and val_loader is not None:
            head.eval()
            _vtot, _vn = 0.0, 0
            with torch.no_grad():
                for _b in val_loader:
                    _w, _z, _sc, _Pt, _rt, _yt, _vt = _b
                    _Pp, _rp, _ = model(_w)
                    _y, _ = _solve_with_algo(args, _Pp, _rp, lp_solver, qp_solver,
                                             agg=agg, sa_tau=sa_tau)
                    _gm = _build_gamma(_w.shape[0], _Pp.shape[1], _w.device, _Pp.dtype,
                                       start_state=_start_state())
                    _v, _ = eval_y_on_true_mdp(_Pt, _rt, _y, _gm, beta)
                    _vtot += float(_v.sum().item()); _vn += int(_v.numel())
            head.train()
            _es_metric = _vtot / max(1, _vn)
        if _es_on and _es_metric > _es_best:
            _es_best = _es_metric
            _es_head = copy.deepcopy(head.state_dict())
            _es_agg = copy.deepcopy(agg.state_dict()) if agg is not None else None
        train_pred_loss_list.append(total_pred_loss / max(1, len(train_loader)))
        train_dfl_loss_list.append(total_dfl_loss / max(1, len(train_loader)))
        train_time_list.append(time.time() - start_time)
        solver_time_sec_per_epoch.append(epoch_solver_time)
        sa_time_sec_per_epoch.append(epoch_sa_time)
        backward_time_sec_per_epoch.append(epoch_backward_time)

        print(f"[Train] epoch {epoch+1}/{args.n_epochs} "
              f"loss={total_loss/max(1,len(train_loader)):.4f} "
              f"slip_CE={train_pred_loss_list[-1]:.4f} "
              f"reward={train_reward_list[-1]:.3f} "
              f"(oracle={train_optimal_reward:.3f})")

        # lightweight checkpoint
        if save_dir is not None:
            ckpt = {
                "epoch": epoch + 1,
                "train_reward_per_epoch": [initial_reward] + train_reward_list,
                "train_pred_loss_per_epoch": train_pred_loss_list,
                "train_dfl_loss_per_epoch": train_dfl_loss_list,
                "train_time": train_time_list,
                "solver_time_sec_per_epoch": solver_time_sec_per_epoch,
                "backward_time_sec_per_epoch": backward_time_sec_per_epoch,
                "algo": args.algo,
            }
            (save_dir / "checkpoint.json").write_text(json.dumps(ckpt, indent=2))
            torch.save(head.state_dict(), save_dir / "head_state.pt")
            torch.save(feature_net.state_dict(), save_dir / "feature_net_state.pt")

    print(f"[Train] done ({time.time()-start_time:.1f}s)")
    if _es_on and _es_head is not None:
        head.load_state_dict(_es_head)
        if agg is not None and _es_agg is not None:
            agg.load_state_dict(_es_agg)
        print(f"[Train] early stop: restored best-train-reward checkpoint ({_es_best:.3f})")

    # --- Evaluation ---------------------------------------------------------------
    print("[Eval] evaluating on test set...")
    head.eval()
    if agg is not None:
        agg.eval()
    test_obj_values: list[float] = []
    test_oracle_values: list[float] = []
    test_slip_ce: list[float] = []
    test_slip_pred_all, test_slip_true_all = [], []
    test_regime_ids: list[int] = []
    test_ambient_slip_chunks: list[np.ndarray] = []
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
    for i, name in enumerate(rnames):
        m = z_test == i
        reg_counts[name] = int(m.sum())
        by_reg[name] = float(r_[m].mean()) if np.any(m) else float("nan")

    result = {
        "train_reward_per_epoch": [initial_reward] + train_reward_list,
        "train_pred_loss_per_epoch": train_pred_loss_list,
        "train_dfl_loss_per_epoch": train_dfl_loss_list,
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
        "train_optimal_reward": train_optimal_reward,
        "algo": args.algo,
    }

    print(f"[Eval] oracle mean = {result['test_mean_oracle_reward']:.3f}")
    print(f"[Eval] pred   mean = {result['test_mean_pred_reward']:.3f}")
    print(f"[Eval] regret mean = {result['test_mean_regret']:.3f}")
    print(f"[Eval] slip CE     = {result['test_slip_ce']:.4f}")

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
    elif args.algo in ("dfl_qpth", "dfl_sketch", "dfl_ssa", "dfl_feas"):
        # At test/metric time all smoothed surrogates (QP regularization,
        # sketching, aggregation) deploy by solving the plain LP on the
        # (predicted) full-state MDP -- the smoothing is a training-gradient
        # device only. Deploying the eta-regularized QP occupancy directly is
        # badly biased at high beta (eta*||y||^2 dominates when ||y|| ~ 1/(1-beta)).
        # dfl_feas is solver-free at train time and likewise deploys the plain LP.
        mdp = CliffWalkingLP(P=P_pred, r=r_pred, beta=beta)
        y_star, _, _, _ = solve_lp(mdp, lp_solver)
    else:
        raise ValueError(f"Unknown algo {args.algo}")
    return y_star, (time.time() - t0)


__all__ = ["pretrain_feature_net", "train_eval_cliff", "eval_y_on_true_mdp"]
