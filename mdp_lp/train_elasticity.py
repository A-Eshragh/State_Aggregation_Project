"""
Training and evaluation for inventory model with elasticity-based demand.

Implements DFL and two-stage methods with price-dependent Poisson demand.
"""

import time
from datetime import datetime
from qpth.qp import QPFunction, QPSolvers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from mdp_lp.elasticity_model import ElasticityModel, LambdaPredictor
from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.utils import (
    solve_lp,
    compute_zero_grad_stats,
    estimate_penalty_lambda_star,
)
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver
from mdp_lp.solvers.differentiable_lp_dual import DifferentiableLPSolverDual
from mdp_lp.solvers.differentiable_qp import DifferentiableQPSolver
from mdp_lp.soft_state_aggregation import SoftAggDense
from mdp_lp.solvers.qpth_qp_solver import solve_occupancy_qp_qpth
from mdp_lp.solvers.sketched_lp import sketched_lp
from mdp_lp.inventory_poisson_grad import poisson_pmf


def _sketched_lp_from_args(H, r, gamma, args, batch_idx=0, epoch=0):
    """Helper: pull sketch hyperparameters from args and call sketched_lp."""
    base_seed = getattr(args, "sketch_seed", None)
    if base_seed is None:
        seed_for_call = None
    else:
        # Vary the seed per (epoch, batch) so sketches stay independent across
        # iterations but reproducible given a fixed --sketch_seed.
        seed_for_call = int(base_seed) + 10007 * int(epoch) + int(batch_idx)
    return sketched_lp(
        H, r, gamma,
        n_sketches=int(getattr(args, "n_sketches", 4)),
        sketch_ratio=float(getattr(args, "sketch_ratio", 0.7)),
        sketch_mode=str(getattr(args, "sketch_mode", "gaussian")),
        sketch_seed=seed_for_call,
        beta=float(getattr(args, "beta", 0.95)),
    )


def prepare_elasticity_data(args):
    """Prepare data for elasticity-based inventory experiments."""
    device = torch.device("cpu")
    args.n_samples = 2 * args.n_samples
    print(f"[Data Prep] Generating {args.n_samples} price scenarios...")
    print(f"[Data Prep] Experiment type: {args.experiment_type}")
    
    if args.experiment_type == 'a':
        p = torch.ones(args.n_samples, device=device) * args.p_constant
        p_related = None
    elif args.experiment_type == 'b':
        if getattr(args, 'fixed_p', False):
            # Manual p: exactly [1, 2, ..., n_samples]. Sample i gets price p_i = i+1. No randomness, no two close p.
            p = torch.arange(1, args.n_samples + 1, dtype=torch.float32, device=device)
            print(f"[Data Prep] Manual p = [1, 2, ..., {args.n_samples}] (one integer price per sample)")
        else:
            p_mean = args.p_lognormal_mean
            p_std = args.p_lognormal_std
            p = torch.exp(torch.randn(args.n_samples, device=device) * p_std + np.log(p_mean))
        p_related = None
    elif args.experiment_type == 'c':
        p_mean = args.p_lognormal_mean
        p_std = args.p_lognormal_std
        p = torch.exp(torch.randn(args.n_samples, device=device) * p_std + np.log(p_mean))
        p_related_mean = args.p_related_lognormal_mean
        p_related_std = args.p_related_lognormal_std
        p_related = torch.exp(torch.randn(args.n_samples, device=device) * p_related_std + np.log(p_related_mean))
    else:
        raise ValueError(f"Unknown experiment_type: {args.experiment_type}")
    
    gt_elasticity = ElasticityModel(
        beta_0=args.beta_0,
        beta_1=args.beta_1,
        beta_2=args.beta_2,
        sigma_epsilon=args.sigma_epsilon
    )
    lambda_true = gt_elasticity(p, p_related)
    # Sanity check: Poisson rate must be strictly positive.
    if not torch.all(torch.isfinite(lambda_true)):
        raise ValueError("lambda_true contains non-finite values (NaN/Inf)")
    if torch.any(lambda_true <= 0):
        min_val = float(lambda_true.min().item())
        raise ValueError(f"lambda_true must be > 0, but min(lambda_true)={min_val}")
    
    solver_obj = DenseLPSolver()
    def lp_solver(H, r, gamma):
        y, basis, dual = solver_obj.solve(H, r, gamma)
        return y, basis, dual

    obj_true_list = []
    y_true_list = []
    dual_true_list = []
    for i in range(args.n_samples):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[Data Prep] Solving MDP {i+1}/{args.n_samples}...")
        mdp = create_inventory_from_lambda(
            lambda_true[i:i+1],
            args.b_max, args.s_max, args.q_max,
            args.foc, args.voc, args.hc, args.bc, args.beta,
            p[i:i+1]
        )
        y_true, obj, dual, _ = solve_lp(mdp, lp_solver) # y_true: (1, S*A)
        obj_true_list.append(obj[0])
        y_true_list.append(y_true)
        dual_true_list.append(dual)

    obj_true = torch.stack(obj_true_list)
    y_true = torch.stack(y_true_list).squeeze(1) # (N, S*A)
    dual_true = torch.stack(dual_true_list).squeeze(1) # (N, S)

    # Prepare features: [p, p_related (if applicable), noise]
    if p_related is not None:
        noise = torch.randn(args.n_samples) * 0.1
        feats = torch.stack([p, p_related, noise], dim=1)
        prices_tensor = torch.stack([p, p_related], dim=1)
    else:
        noise = torch.randn(args.n_samples) * 0.1
        feats = torch.stack([p, noise], dim=1)
        prices_tensor = p
    
    # ZIHAO
    # train_frac = getattr(args, 'train_frac', 0.5)
    # train_size = int(train_frac * args.n_samples) if args.n_samples > 1 else 1
    # test_size = args.n_samples - train_size
    train_size = args.n_samples // 2
    test_size = args.n_samples - train_size
    if args.n_samples == 1:
        train_dataset = TensorDataset(feats, obj_true, lambda_true, prices_tensor, y_true, dual_true)
        test_dataset = TensorDataset(feats, obj_true, lambda_true, prices_tensor, y_true, dual_true)
    else:
        gen = torch.Generator().manual_seed(args.seed)
        train_dataset, test_dataset = random_split(
            TensorDataset(feats, obj_true, lambda_true, prices_tensor, y_true, dual_true),
            [train_size, test_size],
            generator=gen,
        )
    # Zihao
    # if args.mode == 'train':
    #     train_tune_size = int(0.8 * train_size)
    #     val_size = train_size - train_tune_size
    #     train_dataset, test_dataset = random_split(train_dataset, [train_tune_size, val_size])
    
    # Use batching for speed (DFL remains correct; this only reduces Python/optimizer-step overhead).
    #
    # IMPORTANT for qpth/no-aggregation:
    #   The occupancy QP has SA variables where S=(b_max+s_max+1) and A=(q_max+1).
    #   qpth solves dense KKT systems whose cost/memory blows up quickly with SA, especially when batching.
    #   So we intentionally keep batch size very small for dfl_qpth when n_agg=-1.
    default_bs = 128
    # ZIHAO
    # if getattr(args, "algo", None) == "dfl_qpth" and getattr(args, "n_agg", -1) == -1:
    #     # Heuristic: for SA in the thousands (e.g., 151*21=3171), even B=2 can be painful on CPU.
    #     S = int(args.b_max + args.s_max + 1)
    #     A = int(args.q_max + 1)
    #     SA = int(S * A)
    #     default_bs = 1 if SA >= 1500 else 2
    #     print(f"[Data Prep] qpth no-agg detected (SA={SA}). Using small batch_size={default_bs} for feasibility/speed.")

    batch_size = min(default_bs, len(train_dataset)) if len(train_dataset) > 0 else 1
    test_batch_size = min(default_bs, len(test_dataset)) if len(test_dataset) > 0 else 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    print(f"[Data Prep] Data preparation complete")
    
    return train_loader, test_loader, gt_elasticity, p, p_related


# ----------------------------------------
# ZIHAO
def occupancy_to_policy(y_star, S, A, eps=1e-12):
    y = y_star.view(-1, S, A) # clamp_min(0.0)

    d_pred = y.sum(dim=-1)
    pi = y / (d_pred.unsqueeze(-1) + eps)
    return pi, d_pred

# this method leads to small graident
def eval_pi_on_true_mdp(mdp_true, pi, gamma):
    B, S, A = pi.shape
    device, dtype = pi.device, pi.dtype
    beta = mdp_true.beta
    eps = 1e-12

    P = mdp_true.P
    r = mdp_true.r.view(B, S, A)

    P_pi = torch.einsum("bsa,bsan->bsn", pi, P)  # n is next state

    I = torch.eye(S, device=device, dtype=dtype).unsqueeze(0)
    M = I - beta * P_pi.transpose(-1, -2)

    M = M + eps * I

    d = torch.linalg.solve(M, gamma.unsqueeze(-1)).squeeze(-1)

    y = pi * d.unsqueeze(-1)              # (B,S,A)
    r = mdp_true.r.reshape(B, S, A)      # (B,S,A)
    v = (y * r).sum(dim=(1, 2))            # (B,)
    return v, y.reshape(B, S*A)

def eval_y_on_true_mdp(mdp_true, y_star, gamma):
    beta = mdp_true.beta
    P = mdp_true.P
    B, S, A, _ = P.shape
    device, dtype = P.device, P.dtype
    eps = 1e-12
    alpha = 1e-6

    if y_star.dim() == 2:
        y = y_star.view(B, S, A)
    else:
        y = y_star

    d = y.sum(dim=-1, keepdim=True)                     # (B,S,1)
    # pi_star = (y + alpha) / (d + alpha * A + eps)             # (B,S,A)
    pi_star = y / (d + eps)

    P_pi_star = torch.einsum("bsa,bsan->bsn", pi_star, P)           # (B,S,S)
    I = torch.eye(S, device=device, dtype=dtype).unsqueeze(0)
    M = I - beta * P_pi_star.transpose(-1, -2)
    M = M + eps * I

    d_true = torch.linalg.solve(M, gamma.unsqueeze(-1)).squeeze(-1)  # (B,S)

    y_true = pi_star * d_true.unsqueeze(-1)                   # (B,S,A)
    r = mdp_true.r.view(B, S, A)                        # (B,S,A)
    v = (y_true * r).sum(dim=(1, 2))                    # (B,)

    return v, y_true.reshape(B, S * A)

def pretrain_lambda_pred(pred_model, train_loader, pretrain_epochs=60):
    learning_rate = 1e-2
    inner_steps = 20

    optimizer = optim.Adam(pred_model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs)
    pred_loss_fn = nn.MSELoss()

    for epoch in range(pretrain_epochs):
        for batch in train_loader:
            feats, obj_true, lambda_true, prices, y_true, dual_true = batch
            lambda_pred = pred_model(feats)
            loss = pred_loss_fn(lambda_pred, lambda_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pred_model.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()
        print(f"[Warm Start] Epoch {epoch+1}/{pretrain_epochs}, loss: {loss.item()}")
    return pred_model


def pretrain_aggregator(pred_model, agg, train_loader, args, device, pretrain_epochs=20, lr=1e-2):
    """Pre-train the soft-aggregation module to reproduce the oracle occupancy y_true.

    With a random W the aggregated LP → disaggregated y lands on near-arbitrary
    state-action pairs, and the predictor's v*-gradient ends up pointing the
    wrong way (its minimum is reached at ill-posed λ). By first training the
    aggregator to match the *oracle* occupancy under the current (pretrained)
    predictor, W starts from a sensible structure, so the main DFL loss
    becomes monotone in λ_pred (just like plain DFL).

    Predictor is frozen during this phase.
    """
    for p_ in pred_model.parameters():
        p_.requires_grad = False
    for p_ in agg.parameters():
        p_.requires_grad = True

    optimizer = optim.Adam(agg.parameters(), lr=lr)
    solver_obj = DenseLPSolver()
    def lp_solver(H, r, gamma):
        y, basis, dual = solver_obj.solve(H, r, gamma)
        return y, basis, dual

    # λ-augmentation during agg pretraining. With few training samples (small
    # train_size) the aggregator overfits to the oracle λ_true values and
    # learns a W where the LP gives y_disagg ≈ y_true only at those specific
    # points; for any other λ_pred (notably λ → 0) the LP+W still happens to
    # produce a y_disagg that makes v* = r_true·y_disagg look attractive,
    # creating a spurious low-λ local minimum in the main DFL loss. Exposing
    # the aggregator to K random perturbations of λ per step — with fresh
    # oracle y solved on-the-fly — forces W to actually depend on λ, so the
    # main-loop v* landscape has a genuine optimum at λ_pred ≈ λ_true.
    n_lambda_aug = int(getattr(args, "agg_pretrain_lambda_aug", 3))
    aug_scale_min = float(getattr(args, "agg_pretrain_scale_min", 0.3))
    aug_scale_max = float(getattr(args, "agg_pretrain_scale_max", 3.0))
    for epoch in range(pretrain_epochs):
        total = 0.0; n = 0
        for batch in train_loader:
            feats, obj_true, lambda_true, prices, y_true, dual_true = batch
            batch_size = feats.shape[0]
            p_price = prices if prices.dim() == 1 else prices[:, 0]

            # Build the set of λ targets: always include the oracle, then add
            # n_lambda_aug random multiplicative perturbations so the W has
            # to generalize across a range of MDPs.
            lambda_variants = [(lambda_true, y_true)]
            for _ in range(n_lambda_aug):
                scale = (aug_scale_min + (aug_scale_max - aug_scale_min)
                         * torch.rand(batch_size, device=device))
                lambda_k = (lambda_true * scale).clamp_min(0.5)
                with torch.no_grad():
                    mdp_k = create_inventory_from_lambda(
                        lambda_k, args.b_max, args.s_max, args.q_max,
                        args.foc, args.voc, args.hc, args.bc, args.beta, p_price)
                    y_k, _, _, _ = solve_lp(mdp_k, lp_solver)   # (B, S*A)
                lambda_variants.append((lambda_k, y_k))

            batch_loss = 0.0
            for lambda_for_mdp, y_target in lambda_variants:
                mdp = create_inventory_from_lambda(
                    lambda_for_mdp, args.b_max, args.s_max, args.q_max,
                    args.foc, args.voc, args.hc, args.bc, args.beta, p_price)
                S = mdp.P.shape[1]
                gamma = mdp.build_gamma(batch_size, S, device)
                gamma = torch.clamp(gamma, min=1e-8)
                gamma = gamma / gamma.sum(dim=1, keepdim=True)

                P_agg, r_agg, gamma_agg, info = agg(
                    mdp.P, mdp.r, gamma, tau=getattr(args, "sa_tau", 0.5))
                H_agg = mdp.build_H(P_agg)
                r_flat_agg = r_agg.reshape(batch_size, -1)
                y_agg = DifferentiableLPSolver.apply(H_agg, r_flat_agg, gamma_agg)
                y_disagg = SoftAggDense.disaggregate_y(y_agg, info["W"], mdp.r.shape[2])
                if bool(getattr(args, "normalize_disagg", True)):
                    expected_sum = 1.0 / (1.0 - float(args.beta))
                    y_sum = y_disagg.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    y_disagg = y_disagg * (expected_sum / y_sum)
                batch_loss = batch_loss + F.mse_loss(y_disagg, y_target)
            batch_loss = batch_loss / max(1, len(lambda_variants))

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(agg.parameters(), max_norm=10.0)
            optimizer.step()
            total += batch_loss.item() * batch_size; n += batch_size
        print(f"[Agg Pretrain] Epoch {epoch+1}/{pretrain_epochs}, "
              f"avg MSE={total/max(n,1):.4f} ({len(lambda_variants)} λ variants/step)")

    for p_ in pred_model.parameters():
        p_.requires_grad = True
    return agg

def train_eval_elasticity(args, train_loader, test_loader, gt_elasticity):
    """Train and evaluate with elasticity-based inventory model.
    
    Args:
        args: Training arguments
        train_loader: Training data loader
        test_loader: Test data loader
        gt_elasticity: Ground truth elasticity model
    """
    device = torch.device("cpu")
    learning_rate = args.lr
    hidden_dim = args.hidden_dim

    use_p_related = args.experiment_type == 'c'
    
    pred_model = LambdaPredictor(hidden_dim, use_p_related=use_p_related, use_price_only=True).to(device)

    # SoftAggDense init consumes the global RNG and would change DataLoader shuffle vs n_agg=-1.
    # Build it only after pretrain, with RNG save/restore so global RNG matches the no-agg run.
    agg = None
    optimizer = optim.Adam(pred_model.parameters(), lr=learning_rate)
    # Prediction stabilizer loss (used for training signal in both algorithms)
    pred_loss_fn = nn.MSELoss()
    # Objective-difference logging loss: keep scale interpretable (MSE would square thousands → millions)
    obj_diff_fn = nn.L1Loss()
    
    solver_obj = DenseLPSolver()
    def lp_solver(H, r, gamma):
        y, basis, dual = solver_obj.solve(H, r, gamma)
        return y, basis, dual

    train_reward_list = []
    train_lambda_pred_list = []
    train_lambda_true_list = []
    obj_true_list = []
    obj_hat_list = []
    train_time_list = []
    # Extra timing (for best comparisons across solver/aggregation variants)
    sa_time_sec_per_epoch = []         # aggregation wall time per epoch (sum over batches)
    solver_time_sec_per_epoch = []     # LP/QP layer wall time per epoch (sum over batches)
    # For DFL: track MSE (prediction loss) and full DFL loss separately
    train_mse_list = []  # MSE (prediction loss) for DFL
    train_dfl_loss_list = []  # Full DFL loss (-v*_θ + λ·pred_loss)
    # For two-stage: track prediction loss
    train_twostage_loss_list = []  # Prediction loss for two-stage
    # For DFL: gradient magnitudes for weight balance (||∇_θ v*||, ||∇_θ pred_loss||); dℓ/dw = pred_loss
    train_grad_norm_vstar_list = []
    train_grad_norm_pred_loss_list = []
    # For DFL: zero-gradient diagnostics (dV*/dλ_pred)
    train_zero_grad_frac_list = []
    train_dv_dlambda_mean_list = []
    # Policy analysis: track ordering bias per epoch (predicted-lambda policy vs oracle)
    train_order_bias_per_epoch = []
    train_inv_bias_per_epoch = []
    train_lambda_bias_per_epoch = []
    train_frac_over_order_per_epoch = []
    _pa_n_states = args.b_max + args.s_max + 1
    _pa_n_actions = args.q_max + 1
    _pa_action_qty = torch.arange(_pa_n_actions, dtype=torch.float32)
    _pa_inv_level = torch.arange(_pa_n_states, dtype=torch.float32) - args.b_max
    
    print(f"[Training] Starting: {args.n_epochs} epochs, algo={args.algo}")
    start_time = time.time()

    train_obj_true_all = []
    for batch in train_loader:
        _, obj_true, _, _, _, _ = batch
        train_obj_true_all.append(obj_true)
    train_optimal_reward = torch.cat(train_obj_true_all).mean().item()

    qp_solver = QPFunction(eps=1e-10, maxIter=20, solver=QPSolvers.PDIPM_BATCHED)
    checkpoint_every = int(getattr(args, "checkpoint_every", 1))
    save_full_partial = bool(getattr(args, "save_full_partial", True))
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Zihao
    # if args.algo == "dfl_qpth" and getattr(args, "n_agg", -1) == -1:
    #     # qpth no-agg is already very expensive; keep I/O overhead minimal.
    #     checkpoint_every = max(checkpoint_every, 10)
    #     save_full_partial = False

    pred_model.train()
    if args.algo in ("dfl", "dfl_qpth", "dfl_sketch"):
        pretrain_lambda_pred(pred_model, train_loader, pretrain_epochs=getattr(args, "pretrain_epochs", 30))

    if getattr(args, "n_agg", -1) != -1 and args.algo in ("dfl", "dfl_qpth", "dfl_sketch"):
        n_states = args.b_max + args.s_max + 1
        n_actions = args.q_max + 1
        _rng_state = torch.get_rng_state()
        torch.manual_seed(args.seed + int(args.n_agg))
        agg = SoftAggDense(
            n_states=n_states,
            n_actions=n_actions,
            n_clusters=args.n_agg,
            r_embed=getattr(args, "sa_r_embed", 8),
            fixed_W=bool(getattr(args, "agg_fixed_W", False)),
            fixed_W_seed=int(getattr(args, "agg_fixed_W_seed", 1)),
        ).to(device)
        torch.set_rng_state(_rng_state)
        # NOTE: --agg_lr_ratio is documented as "Aggregator LR as fraction of
        # predictor LR (two-timescale training)". Earlier this was inverted —
        # the ratio was applied to the predictor, so with the default
        # agg_lr_ratio=0.01 the predictor learned at 1/100 the aggregator's
        # rate and collapsed to a near-constant while W absorbed all input
        # variation. The predictor now keeps `learning_rate`, the aggregator
        # is scaled by agg_lr_ratio (matching the documented intent).
        agg_lr_ratio = float(getattr(args, "agg_lr_ratio", 0.01))
        optimizer = optim.Adam([
            {"params": pred_model.parameters(), "lr": learning_rate},
            {"params": agg.parameters(),        "lr": learning_rate * agg_lr_ratio},
        ])

        agg_pretrain_epochs = int(getattr(args, "agg_pretrain_epochs", 0))
        if agg_pretrain_epochs > 0:
            pretrain_aggregator(pred_model, agg, train_loader, args, device,
                                pretrain_epochs=agg_pretrain_epochs)

    params = list(pred_model.parameters())
    if agg is not None:
        params.extend(list(agg.parameters()))

    # Epoch 0: same metric as epochs 1+ for both DFL and twostage — (r_true · y_star) at init, so they start at the same point.
    pred_model.eval()
    initial_rewards = []
    with torch.no_grad():
        for batch_data in train_loader:
            feats, obj_true, lambda_true, prices, y_true, dual_true = batch_data
            batch_size = feats.shape[0]
            lambda_pred = pred_model(feats)
            
            if use_p_related:
                p = prices[:, 0]
            else:
                p = prices if prices.dim() == 1 else prices

            if torch.any(lambda_pred < 0):
                lambda_pred = torch.zeros_like(lambda_pred)

            mdp = create_inventory_from_lambda(
                lambda_pred, args.b_max, args.s_max, args.q_max,
                args.foc, args.voc, args.hc, args.bc, args.beta, p
            )
            mdp_true = create_inventory_from_lambda(
                lambda_true, args.b_max, args.s_max, args.q_max,
                args.foc, args.voc, args.hc, args.bc, args.beta, p
            )
            S = mdp.P.shape[1]
            gamma = mdp.build_gamma(batch_size, S, device)
            gamma = torch.clamp(gamma, min=1e-8)
            gamma = gamma / gamma.sum(dim=1, keepdim=True)
            gamma_use = gamma
            if args.algo in ("dfl", "dfl_qpth", "dfl_sketch"):
                P_use, r_use, gamma_use = mdp.P, mdp.r, gamma
                _agg_info = None
                if agg is not None:
                    P_use, r_use, gamma_agg, _agg_info = agg(P_use, r_use, gamma_use, tau=getattr(args, "sa_tau", 0.5))
                H_use = mdp.build_H(P_use)
                r_flat_use = r_use.reshape(batch_size, -1)
                if args.algo == "dfl":
                    if agg is not None and _agg_info is not None:
                        # y_star = DifferentiableLPSolver.apply(H_use, r_flat_use, gamma_agg)
                        y_star, dual = DifferentiableLPSolverDual.apply(H_use, r_flat_use, gamma_agg)
                    else:
                        # y_star = DifferentiableLPSolver.apply(H_use, r_flat_use, gamma_use)
                        y_star, dual = DifferentiableLPSolverDual.apply(H_use, r_flat_use, gamma_use)
                elif args.algo == "dfl_sketch":
                    # Initial-pass v* reporting: use the *plain* LP solution
                    # (the sketched y_avg is a stochastic mix of vertices and
                    # rolls out badly on the true MDP at degenerate cells).
                    if agg is not None and _agg_info is not None:
                        y_plain, _, _ = lp_solver(H_use, r_flat_use, gamma_agg)
                    else:
                        y_plain, _, _ = lp_solver(H_use, r_flat_use, gamma_use)
                    y_star = y_plain
                else:
                    if agg is not None and _agg_info is not None:
                        y_star = solve_occupancy_qp_qpth(
                            H_use, r_flat_use, gamma_agg,
                            reg=getattr(args, "qp_reg", 0.1), qp_solver=qp_solver
                        )
                    else:
                        y_star = solve_occupancy_qp_qpth(
                            H_use, r_flat_use, gamma_use,
                            reg=getattr(args, "qp_reg", 0.1), qp_solver=qp_solver
                        )
                if agg is not None and _agg_info is not None:
                    y_star = SoftAggDense.disaggregate_y(y_star, _agg_info["W"], mdp.r.shape[2])
            else:
                y_star, _, _, _ = solve_lp(mdp, lp_solver)

            with torch.no_grad():
                v_star_init, _ = eval_y_on_true_mdp(mdp_true, y_star, gamma_use)
            # r_true = mdp_true.r.reshape(batch_size, -1)
            # v_star_init = (r_true * y_star).sum(dim=1).mean().item()
            initial_rewards.extend(v_star_init.flatten().tolist())
    initial_reward = float(np.mean(initial_rewards))

    pred_model.train()
    # Respect --agg_warmup_epochs so the aggregator can be frozen for the first
    # N epochs (anti-shortcut: forces the predictor to take the first DFL
    # gradient steps before the aggregator can absorb the input variation).
    agg_warmup_epochs = int(getattr(args, "agg_warmup_epochs", 0))
    fidelity_weight = float(getattr(args, "fidelity_weight", 0.0))

    for epoch in range(args.n_epochs):
        if agg is not None:
            freeze_agg = (epoch < agg_warmup_epochs)
            for p_ in agg.parameters():
                p_.requires_grad = (not freeze_agg)
        # Track estimated stationary point lambda* for the penalty Lagrangian loss.
        epoch_lambda_star_mean = []
        epoch_lambda_star_min = []
        epoch_lambda_star_max = []
        epoch_lambda_star_grad_abs = []
        epoch_lambda_star_last_batch = None
        train_loss = 0.0
        obj_loss = 0.0
        epoch_sa_time = 0.0
        epoch_solver_time = 0.0
        epoch_rewards = []
        epoch_lambda_pred = []
        epoch_lambda_true = []
        epoch_mse = []  # For DFL: track MSE per batch
        epoch_dfl_loss = []  # For DFL: track full loss per batch
        epoch_vstar = []  # For DFL: v* (reward) per batch
        epoch_aug_lag = []  # For DFL: constraint penalty per batch
        epoch_twostage_loss = []  # For two-stage: track prediction loss per batch
        epoch_grad_norm_vstar = []  # For DFL: ||∇_θ v*|| per batch
        epoch_grad_norm_pred_loss = []  # For DFL: ||∇_θ pred_loss|| per batch
        epoch_zero_grad_frac = []  # For DFL: zero-grad fraction per batch
        epoch_dv_dlambda_mean = []  # For DFL: mean |dV*/dλ_pred| per batch
        epoch_order_pred = []   # policy analysis: E[order] under predicted-λ policy
        epoch_order_oracle = [] # policy analysis: E[order] under oracle (true-λ) policy
        epoch_inv_pred = []     # policy analysis: E[inventory] under predicted-λ policy
        epoch_inv_oracle = []   # policy analysis: E[inventory] under oracle policy
        
        for batch_idx, batch_data in enumerate(train_loader):
            feats, obj_true, lambda_true, prices, y_true, dual_true = batch_data
            batch_size = feats.shape[0]

            lambda_pred = pred_model(feats)
            if not torch.all(torch.isfinite(lambda_pred)):
                raise ValueError("lambda_pred contains non-finite values (NaN/Inf)")

            if use_p_related:
                p = prices[:, 0]
            else:
                p = prices if prices.dim() == 1 else prices
            
            if torch.any(lambda_pred < 1):
                lambda_pred_pos = torch.clamp(lambda_pred, min=1.0)
            else:
                lambda_pred_pos = lambda_pred

            mdp = create_inventory_from_lambda(
                lambda_pred_pos,
                args.b_max, args.s_max, args.q_max,
                args.foc, args.voc, args.hc, args.bc, args.beta,
                p
            )
            
            S = mdp.P.shape[1]
            gamma = mdp.build_gamma(batch_size, S, device)
            gamma = torch.clamp(gamma, min=1e-8)
            gamma = gamma / gamma.sum(dim=1, keepdim=True)
            gamma_use = gamma

            if args.algo in ("dfl", "dfl_qpth", "dfl_sketch"):
                # Optionally aggregate (P, r, gamma) into M=n_agg clusters.
                P_use, r_use, gamma_use = mdp.P, mdp.r, gamma
                _agg_info = None
                if agg is not None:
                    P_use, r_use, gamma_agg, _agg_info = agg(
                        P_use, r_use, gamma_use, tau=getattr(args, "sa_tau", 0.5)
                    )
                    epoch_sa_time += float(_agg_info.get("runtime_sec", 0.0))

                H_use = mdp.build_H(P_use)
                r_flat_use = r_use.reshape(batch_size, -1)

                if args.algo == "dfl":
                    # Differentiable LP via fixed-basis sensitivity (uses Gurobi to get a basis).
                    t0 = time.time()
                    if agg is not None and _agg_info is not None:
                        # y_star = DifferentiableLPSolver.apply(H_use, r_flat_use, gamma_agg)
                        y_star, dual = DifferentiableLPSolverDual.apply(H_use, r_flat_use, gamma_agg)
                    else:
                        # y_star = DifferentiableLPSolver.apply(H_use, r_flat_use, gamma_use)
                        y_star, dual = DifferentiableLPSolverDual.apply(H_use, r_flat_use, gamma_use)
                    epoch_solver_time += (time.time() - t0)
                elif args.algo == "dfl_sketch":
                    # Sketch-and-solve: ensemble of row-sketched LPs averaged for smoothness.
                    t0 = time.time()
                    if agg is not None and _agg_info is not None:
                        y_star = _sketched_lp_from_args(H_use, r_flat_use, gamma_agg, args,
                                                        batch_idx=batch_idx, epoch=epoch)
                    else:
                        y_star = _sketched_lp_from_args(H_use, r_flat_use, gamma_use, args,
                                                        batch_idx=batch_idx, epoch=epoch)
                    epoch_solver_time += (time.time() - t0)
                elif args.algo == "dfl_qpth":
                    t0 = time.time()
                    if agg is not None and _agg_info is not None:
                        y_star = solve_occupancy_qp_qpth(
                            H_use, r_flat_use, gamma_agg,
                            reg=getattr(args, "qp_reg", 0.1),
                            qp_solver=qp_solver
                        )
                    else:
                        y_star = solve_occupancy_qp_qpth(
                            H_use, r_flat_use, gamma_use,
                            reg=getattr(args, "qp_reg", 0.1),
                            qp_solver=qp_solver
                        )
                    epoch_solver_time += (time.time() - t0)
                else:
                    raise ValueError(f"Invalid algorithm: {args.algo}")

                with torch.no_grad():
                    mdp_true = create_inventory_from_lambda(lambda_true, args.b_max, args.s_max, args.q_max,
                                                            args.foc, args.voc, args.hc, args.bc, args.beta,
                                                            p)
                    r_true = mdp_true.r.reshape(batch_size, -1)
                    H_true = mdp_true.build_H(mdp_true.P)

                if agg is not None and _agg_info is not None:
                    W = _agg_info["W"]                                    # (B, S, M)
                    mass = W.sum(dim=1).clamp_min(1e-8)                   # (B, M)
                    r_true_3d = mdp_true.r                                # (B, S, A)
                    r_true_agg = torch.einsum(
                        "bsm,bsa->bma", W, r_true_3d
                    ) / mass.unsqueeze(-1)                                # (B, M, A)
                    right = torch.einsum("bsaj,bjm->bsam", mdp_true.P, W) # (B, S, A, M)
                    p_hat = torch.einsum("bsm,bsan->bman", W, right)      # (B, M, A, M)
                    P_true_agg = p_hat / mass[:, :, None, None]
                    P_true_agg = P_true_agg.clamp_min(0.0)
                    P_true_agg = P_true_agg / P_true_agg.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                    gamma_true_agg = torch.einsum("bs,bsm->bm", gamma_use, W).clamp_min(1e-8)
                    gamma_true_agg = gamma_true_agg / gamma_true_agg.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    H_true_agg = mdp_true.build_H(P_true_agg)             # (B, M, M*A)
                    if args.algo == "dfl_sketch":
                        # Keep y_star from the PRED-aggregated sketched solve
                        # above. Re-solving the LP in TRUE-projected space here
                        # kills the direct λ_pred → y_star gradient — only W
                        # would carry the signal, and W alone can absorb all
                        # input variation, causing the predictor to collapse
                        # to a near-constant. Evaluating the PRED y_star
                        # against TRUE-projected (r,γ,H) matches the
                        # formulation diagnose_solver_gradients and
                        # compute_loss_grad already use.
                        pass
                    else:
                        # y_star = DifferentiableLPSolver.apply(H_true_agg, r_true_agg.reshape(batch_size, -1), gamma_true_agg)
                        y_star, dual = DifferentiableLPSolverDual.apply(H_true_agg, r_true_agg.reshape(batch_size, -1), gamma_true_agg)

                    violation = gamma_true_agg - (H_true_agg @ y_star.unsqueeze(-1)).squeeze(-1)
                    rho = getattr(args, 'aug_lag_rho', 0.1)
                    aug_lag = rho * violation.pow(2).mean()
                    v_star = torch.sum(r_true_agg.reshape(batch_size, -1) * y_star, dim=1)
                    dfl_loss = -v_star.mean() + aug_lag
                else:
                    violation = gamma_use - (H_true @ y_star.unsqueeze(-1)).squeeze(-1)
                    rho = getattr(args, 'aug_lag_rho', 0.1)
                    aug_lag = rho * violation.pow(2).mean()
                    v_star = torch.sum(r_true * y_star, dim=1)
                    dfl_loss = -v_star.mean() + aug_lag

                # 3. Optional fidelity loss: anchor disaggregated occupancy to
                # the oracle y_true. Gives the aggregator a supervised target
                # that does not require lambda_pred to be wrong.
                if fidelity_weight > 0:
                    fidelity_loss = F.mse_loss(y_star, y_true)
                    dfl_loss = dfl_loss + fidelity_weight * fidelity_loss

                # 2. True evaluation loss
                # v_star, _y_pi_true = eval_y_on_true_mdp(mdp_true, y_star, gamma_use, W=W_agg)
                # dfl_loss = -v_star.mean() * 10000
                # aug_lag = torch.tensor(0.0)

                lambda_star_info = estimate_penalty_lambda_star(
                    dfl_loss=dfl_loss,
                    lambda_pred=lambda_pred,
                    lambda_min=1.0,
                    curvature_eps=1e-6,
                    max_newton_step=10.0,
                )
                if lambda_star_info is not None:
                    epoch_lambda_star_mean.append(lambda_star_info["lambda_star_mean"])
                    epoch_lambda_star_min.append(lambda_star_info["lambda_star_min"])
                    epoch_lambda_star_max.append(lambda_star_info["lambda_star_max"])
                    epoch_lambda_star_grad_abs.append(lambda_star_info["grad_abs_mean"])
                    epoch_lambda_star_last_batch = np.round(
                        np.asarray(lambda_star_info["lambda_star_list"], dtype=float), 3
                    ).tolist()

                # Zero-grad diagnostic: dV*/dλ_pred
                # zg_frac, dv_dl_mean, _ = compute_zero_grad_stats(v_star, lambda_pred)
                zg_frac = 0.0
                dv_dl_mean = 0.0
                epoch_zero_grad_frac.append(zg_frac)
                epoch_dv_dlambda_mean.append(dv_dl_mean)

                pred_loss = pred_loss_fn(lambda_pred, lambda_true)

                def _grad_norm_from_grads(grads):
                    total = sum(g.pow(2).sum().item() for g in grads if g is not None)
                    return float(np.sqrt(total)) if total > 0 else 0.0

                params_list = list(pred_model.parameters())
                grads_v = torch.autograd.grad(dfl_loss.mean(), params_list,
                                              retain_graph=True, allow_unused=True)
                g_v_star = _grad_norm_from_grads(grads_v)
                epoch_grad_norm_vstar.append(g_v_star)

                grads_p = torch.autograd.grad(pred_loss, params_list,
                                              retain_graph=True, allow_unused=True)
                g_pred_loss = _grad_norm_from_grads(grads_p)
                epoch_grad_norm_pred_loss.append(g_pred_loss)

                dfl_loss_final = dfl_loss + args.loss_weight * pred_loss
                optimizer.zero_grad()
                dfl_loss_final.backward()
                lambda_true_sorted = np.sort(lambda_true.detach().cpu().reshape(-1).numpy())
                lambda_pred_sorted = np.sort(lambda_pred.detach().cpu().reshape(-1).numpy())
                print(f"lambda_true(sorted): {lambda_true_sorted}")
                print(f"lambda_pred(sorted): {lambda_pred_sorted}")
                if epoch_lambda_star_last_batch is not None:
                    lambda_star_sorted = sorted(epoch_lambda_star_last_batch)
                    print(f"lambda_star(sorted): {lambda_star_sorted}")

                torch.nn.utils.clip_grad_norm_(params, max_norm=getattr(args, "grad_clip", 10.0))

                epoch_dfl_loss.append(dfl_loss_final.item())
                train_loss += dfl_loss_final.item()

                obj_hat = v_star.detach()
                obj_diff = obj_diff_fn(obj_hat, obj_true)
                obj_loss += obj_diff.item()

                epoch_mse.append(pred_loss.item())
                epoch_vstar.append(v_star.mean().item())
                epoch_aug_lag.append(aug_lag.item())

                epoch_lambda_pred.append(lambda_pred.detach())
                epoch_lambda_true.append(lambda_true)
            else: # two-stage
                pred_loss = pred_loss_fn(lambda_pred, lambda_true)
                
                # with torch.no_grad():
                #     y_star, obj_hat, _, _ = solve_lp(mdp, lp_solver)
                #     obj_hat = obj_hat.detach()
                #     obj_diff = obj_diff_fn(obj_hat, obj_true)
                #     obj_loss += obj_diff.item()

                obj_hat = torch.tensor(99999)
                    
                epoch_twostage_loss.append(pred_loss.item())
                epoch_lambda_pred.append(lambda_pred.detach())
                epoch_lambda_true.append(lambda_true)

                loss = pred_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=getattr(args, "grad_clip", 10.0))
                train_loss += loss.item()

            optimizer.step()

        with torch.no_grad():
            mdp_true = create_inventory_from_lambda(lambda_true, args.b_max, args.s_max, args.q_max,
                                                    args.foc, args.voc, args.hc, args.bc, args.beta,
                                                    p)
            if args.algo == "dfl_sketch":
                # Sketched y_avg is a smoothed inner-product proxy for the
                # *training gradient* only. At degenerate LP cells (e.g. low λ)
                # the K sketches snap to different near-tied vertices and y_avg
                # becomes a smear over disagreeing deterministic policies; the
                # extracted policy (soft or greedy) rolls out terribly on the
                # true MDP. For v* reporting we evaluate the deployable policy
                # by re-solving the *plain* LP at λ_pred.
                y_plain_pred, _, _ = lp_solver(H_use, r_flat_use,
                                               gamma_agg if (agg is not None and _agg_info is not None) else gamma_use)
                if agg is not None and _agg_info is not None:
                    y_star_eval = SoftAggDense.disaggregate_y(
                        y_plain_pred, _agg_info["W"], mdp.r.shape[2])
                else:
                    y_star_eval = y_plain_pred
            elif agg is not None and _agg_info is not None:
                y_star_eval = SoftAggDense.disaggregate_y(y_star, _agg_info["W"], mdp.r.shape[2])
            else:
                y_star_eval = y_star
            v_eval, _ = eval_y_on_true_mdp(mdp_true, y_star_eval, gamma_use)
            v_star = v_eval.mean().item()
            epoch_rewards.append(v_star)
        
        train_reward_list.append(float(np.mean(epoch_rewards)))
        train_time_list.append(time.time() - start_time)
        sa_time_sec_per_epoch.append(epoch_sa_time)
        solver_time_sec_per_epoch.append(epoch_solver_time)
        avg_obj_loss = obj_loss / len(train_loader)
        train_lambda_pred_list.append(torch.cat(epoch_lambda_pred))
        train_lambda_true_list.append(torch.cat(epoch_lambda_true))
        obj_hat_list.append(obj_hat)

        
        # For DFL variants (LP-based and qpth-based): track average MSE and full DFL loss per epoch
        if args.algo in ('dfl', 'dfl_qpth', 'dfl_sketch') and epoch_mse:
            train_mse_list.append(np.mean(epoch_mse))
            train_dfl_loss_list.append(np.mean(epoch_dfl_loss))
            train_twostage_loss_list.append(0.0)  # Placeholder for two-stage
            train_grad_norm_vstar_list.append(np.mean(epoch_grad_norm_vstar) if epoch_grad_norm_vstar else 0.0)
            train_grad_norm_pred_loss_list.append(np.mean(epoch_grad_norm_pred_loss) if epoch_grad_norm_pred_loss else 0.0)
            train_zero_grad_frac_list.append(np.mean(epoch_zero_grad_frac) if epoch_zero_grad_frac else 0.0)
            train_dv_dlambda_mean_list.append(np.mean(epoch_dv_dlambda_mean) if epoch_dv_dlambda_mean else 0.0)
        else:
            # Two-stage (or any other non-DFL mode): no DFL loss/MSE
            train_mse_list.append(0.0)
            train_dfl_loss_list.append(0.0)
            train_grad_norm_vstar_list.append(0.0)
            train_grad_norm_pred_loss_list.append(0.0)
            train_zero_grad_frac_list.append(0.0)
            train_dv_dlambda_mean_list.append(0.0)
            # For two-stage: track prediction loss
            if epoch_twostage_loss:
                train_twostage_loss_list.append(np.mean(epoch_twostage_loss))
            else:
                train_twostage_loss_list.append(0.0)
        
        # Policy analysis: aggregate per-epoch (or placeholders when --policy_analysis not set)
        _ep_lam_pred = torch.cat(epoch_lambda_pred)
        _ep_lam_true = torch.cat(epoch_lambda_true)
        train_lambda_bias_per_epoch.append(float((_ep_lam_pred - _ep_lam_true).abs().mean()))
        if getattr(args, 'policy_analysis', False) and epoch_order_pred:
            _ep_ord_pred = torch.cat(epoch_order_pred)
            _ep_ord_oracle = torch.cat(epoch_order_oracle)
            _ep_inv_pred = torch.cat(epoch_inv_pred)
            _ep_inv_oracle = torch.cat(epoch_inv_oracle)
            _ep_ord_bias = _ep_ord_pred - _ep_ord_oracle
            train_order_bias_per_epoch.append(float(_ep_ord_bias.abs().mean()))
            train_inv_bias_per_epoch.append(float((_ep_inv_pred - _ep_inv_oracle).abs().mean()))
            train_frac_over_order_per_epoch.append(float((_ep_ord_bias > 0).float().mean()))
        else:
            train_order_bias_per_epoch.append(0.0)
            train_inv_bias_per_epoch.append(0.0)
            train_frac_over_order_per_epoch.append(0.0)
        
        if (epoch + 1) % 1 == 0:
            msg = f'[Training] Epoch {epoch + 1}/{args.n_epochs}, Loss: {train_loss / len(train_loader):.4f}, λ_bias={train_lambda_bias_per_epoch[-1]:+.3f}'
            if args.algo in ('dfl', 'dfl_qpth', 'dfl_sketch') and epoch_vstar and epoch_aug_lag and epoch_mse:
                msg += f'  |  v*={epoch_rewards[-1]:.2f}  aug_lag={epoch_aug_lag[-1]:.2f}  pred_MSE={epoch_mse[-1]:.4f}'
                msg += f'  zero_grad={train_zero_grad_frac_list[-1]:.1%}  |dV/dλ|={train_dv_dlambda_mean_list[-1]:.4f}'
            if epoch_lambda_star_mean:
                msg += (f"  |  λ*≈{np.mean(epoch_lambda_star_mean):.3f}"
                        f" [{np.mean(epoch_lambda_star_min):.3f}, {np.mean(epoch_lambda_star_max):.3f}]"
                        f"  |dL/dλ|={np.mean(epoch_lambda_star_grad_abs):.4e}")
            if getattr(args, 'policy_analysis', False):
                msg += f', order_bias={train_order_bias_per_epoch[-1]:+.4f}, frac_over_order={train_frac_over_order_per_epoch[-1]:.1%}'
            print(msg)
        
        should_checkpoint = ((epoch + 1) % checkpoint_every == 0) or (epoch == 0) or ((epoch + 1) == args.n_epochs)
        if should_checkpoint and hasattr(args, 'save_dir') and args.save_dir:
            checkpoint_file = args.save_dir / 'checkpoint.json'
            model_file = args.save_dir / 'model_state.pt'
            
            results_partial_file = args.save_dir / f'results_partial_{run_timestamp}.json'
            try:
                import json
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'train_reward_per_epoch': [float(initial_reward)] + [float(r) for r in train_reward_list],
                    'train_time': train_time_list.copy(),
                    'sa_time_sec_per_epoch': sa_time_sec_per_epoch.copy(),
                    'solver_time_sec_per_epoch': solver_time_sec_per_epoch.copy(),
                }
                checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2))
                torch.save(pred_model.state_dict(), model_file)

                # Avoid rewriting gigantic arrays every checkpoint unless explicitly requested.
                partial_results = {
                    'epoch': epoch + 1,
                    'n_epochs': args.n_epochs,
                    'status': 'training_in_progress',
                    'train_reward_per_epoch': [float(initial_reward)] + [float(r) for r in train_reward_list],
                    'train_mse_per_epoch': [float(x) for x in train_mse_list],
                    'train_dfl_loss_per_epoch': [float(x) for x in train_dfl_loss_list],
                    'train_twostage_loss_per_epoch': [float(x) for x in train_twostage_loss_list],
                    'train_grad_norm_vstar_per_epoch': [float(x) for x in train_grad_norm_vstar_list],
                    'train_grad_norm_pred_loss_per_epoch': [float(x) for x in train_grad_norm_pred_loss_list],
                    'train_zero_grad_frac_per_epoch': [float(x) for x in train_zero_grad_frac_list],
                    'train_dv_dlambda_mean_per_epoch': [float(x) for x in train_dv_dlambda_mean_list],
                    'train_time': train_time_list.copy(),
                    'sa_time_sec_per_epoch': sa_time_sec_per_epoch.copy(),
                    'solver_time_sec_per_epoch': solver_time_sec_per_epoch.copy(),
                    'test_obj_values': None,
                    'test_lambda_pred': None,
                    'test_lambda_true': None,
                    'train_order_bias_per_epoch': train_order_bias_per_epoch.copy(),
                    'train_inv_bias_per_epoch': train_inv_bias_per_epoch.copy(),
                    'train_lambda_bias_per_epoch': train_lambda_bias_per_epoch.copy(),
                    'train_frac_over_order_per_epoch': train_frac_over_order_per_epoch.copy(),
                    'obj_true': [obj.cpu().numpy().tolist() for obj in obj_true_list],
                    'obj_hat': [obj.cpu().numpy().tolist() for obj in obj_hat_list]
                }
                if save_full_partial:
                    partial_results['train_lambda_pred'] = [l.cpu().numpy().tolist() for l in train_lambda_pred_list]
                    partial_results['train_lambda_true'] = [l.cpu().numpy().tolist() for l in train_lambda_true_list]
                results_partial_file.write_text(json.dumps(partial_results, indent=2))
            except Exception as e:
                print(f'[Training] ERROR saving checkpoint: {e}')
                import traceback
                traceback.print_exc()
                raise

    train_reward_list = [initial_reward] + train_reward_list
    
    print(f"[Training] Complete. Time: {time.time() - start_time:.2f}s")
    
    if hasattr(args, 'save_dir') and args.save_dir:
        try:
            final_checkpoint = args.save_dir / 'checkpoint_final.json'
            final_model = args.save_dir / 'model_state_final.pt'
            import json
            checkpoint_data = {
                'epoch': args.n_epochs,
                'train_reward_per_epoch': [float(r) for r in train_reward_list],
                'train_time': train_time_list.copy(),
                'sa_time_sec_per_epoch': sa_time_sec_per_epoch.copy(),
                'solver_time_sec_per_epoch': solver_time_sec_per_epoch.copy(),
                'train_lambda_pred': [l.cpu().numpy().tolist() for l in train_lambda_pred_list],
                'train_lambda_true': [l.cpu().numpy().tolist() for l in train_lambda_true_list],
                'obj_true': [obj.cpu().numpy().tolist() for obj in obj_true_list],
                'obj_hat': [obj.cpu().numpy().tolist() for obj in obj_hat_list]
            }
            final_checkpoint.write_text(json.dumps(checkpoint_data, indent=2))
            torch.save(pred_model.state_dict(), final_model)

            # Save SoftAggDense
            if agg is not None and int(getattr(args, "n_agg", -1)) != -1:
                agg_final_path = args.save_dir / 'agg_state_final.pt'
                agg_payload = {
                    'state_dict': agg.state_dict(),
                    'n_states': int(args.b_max + args.s_max + 1),
                    'n_actions': int(args.q_max + 1),
                    'n_clusters': int(args.n_agg),
                    'r_embed': int(getattr(args, 'sa_r_embed', 8)),
                    'sa_tau': float(getattr(args, 'sa_tau', 0.5)),
                    'seed': int(args.seed),
                }
                torch.save(agg_payload, agg_final_path)
                print(f'[Training] Saved SoftAggDense to {agg_final_path}')
        except Exception as e:
            print(f'[Training] ERROR saving final checkpoint: {e}')
            import traceback
            traceback.print_exc()
            raise
    
    print(f"[Evaluation] Starting evaluation...")
    pred_model.eval()
    test_obj_values = []       # v(y_pred) on true MDP — reward from predicted lambda
    test_oracle_values = []    # v(y_oracle) on true MDP — reward from true lambda (upper bound)
    test_lambda_pred = []
    test_lambda_true = []
    test_obj_true = []
    test_dual_hat = []
    test_dual_oracle = []
    test_lambda_mse = []       # per-batch MSE(lambda_pred, lambda_true)
    with torch.no_grad():
        for batch_data in test_loader:
            feats, obj_true, lambda_true, prices, y_true, dual_true = batch_data
            batch_size = feats.shape[0]

            lambda_pred = pred_model(feats)
            if not torch.all(torch.isfinite(lambda_pred)):
                raise ValueError("lambda_pred contains non-finite values (NaN/Inf) during eval")
            # Match training-time handling (lines ~550-553): the Poisson rate must be
            # strictly positive for `create_inventory_from_lambda`, but the unconstrained
            # MLP head can occasionally emit small/negative values. Clamp for MDP
            # construction while keeping the raw prediction for MSE logging.
            if torch.any(lambda_pred < 1):
                min_val = float(lambda_pred.min().item())
                print(f"[Evaluation] WARNING: min(lambda_pred)={min_val:.4f} < 1; clamping to 1.0 for MDP construction.")
                lambda_pred_pos = torch.clamp(lambda_pred, min=1.0)
            else:
                lambda_pred_pos = lambda_pred

            if use_p_related:
                p = prices[:, 0]
            else:
                p = prices if prices.dim() == 1 else prices

            # --- Oracle: solve with true lambda ---
            mdp_true_test = create_inventory_from_lambda(
                lambda_true,
                args.b_max, args.s_max, args.q_max,    
                args.foc, args.voc, args.hc, args.bc, args.beta,
                p
            )
            S_test = mdp_true_test.P.shape[1]
            gamma_test = mdp_true_test.build_gamma(batch_size, S_test, lambda_true.device)
            gamma_test = torch.clamp(gamma_test, min=1e-8)
            gamma_test = gamma_test / gamma_test.sum(dim=1, keepdim=True)

            y_oracle, obj_oracle, dual_oracle, _ = solve_lp(mdp_true_test, lp_solver)
            v_oracle, _ = eval_y_on_true_mdp(mdp_true_test, y_oracle, gamma_test)
            test_oracle_values.extend(v_oracle.tolist())
            test_dual_oracle.append(dual_oracle)

            # --- Predicted: solve with predicted lambda, evaluate on true MDP ---
            mdp_pred = create_inventory_from_lambda(
                lambda_pred_pos,
                args.b_max, args.s_max, args.q_max,
                args.foc, args.voc, args.hc, args.bc, args.beta,
                p
            )
            y_star_test, obj_hat, dual_hat, _ = solve_lp(mdp_pred, lp_solver)
            v_pred, _ = eval_y_on_true_mdp(mdp_true_test, y_star_test, gamma_test)
            test_obj_values.extend(v_pred.tolist())

            # --- Per-batch lambda MSE ---
            batch_mse = F.mse_loss(lambda_pred, lambda_true).item()
            test_lambda_mse.append(batch_mse)

            test_lambda_pred.append(lambda_pred)
            test_lambda_true.append(lambda_true)
            test_obj_true.append(obj_true)
            test_dual_hat.append(dual_hat)

    test_obj_values = torch.tensor(test_obj_values)
    test_oracle_values = torch.tensor(test_oracle_values)
    test_lambda_pred = torch.cat(test_lambda_pred)
    test_lambda_true = torch.cat(test_lambda_true)
    test_dual_hat = torch.cat(test_dual_hat)
    test_dual_oracle = torch.cat(test_dual_oracle)

    # --- Summary metrics for comparison ---
    test_regret = test_oracle_values - test_obj_values  # per-sample regret (>= 0 ideally)
    test_mse = F.mse_loss(test_lambda_pred, test_lambda_true).item()
    test_mean_pred_reward = test_obj_values.mean().item()
    test_mean_oracle_reward = test_oracle_values.mean().item()
    test_mean_regret = test_regret.mean().item()

    print(f"[Evaluation] Oracle reward (true λ):     {test_mean_oracle_reward:.4f}")
    print(f"[Evaluation] Predicted reward (pred λ):   {test_mean_pred_reward:.4f}")
    print(f"[Evaluation] Mean regret:                 {test_mean_regret:.4f}")
    print(f"[Evaluation] Lambda MSE:                  {test_mse:.6f}")
    print(f"[Evaluation] Algo: {args.algo}")

    if hasattr(args, 'save_dir') and args.save_dir:
        try:
            eval_backup = args.save_dir / 'evaluation_backup.json'
            import json
            eval_data = {
                'test_obj_true': torch.cat(test_obj_true).cpu().numpy().tolist(),
                'test_obj_values': test_obj_values.cpu().numpy().tolist(),
                'test_oracle_values': test_oracle_values.cpu().numpy().tolist(),
                'test_regret': test_regret.cpu().numpy().tolist(),
                'test_lambda_pred': test_lambda_pred.cpu().numpy().tolist(),
                'test_lambda_true': test_lambda_true.cpu().numpy().tolist(),
                'test_dual_hat': test_dual_hat.cpu().numpy().tolist(),
                'test_dual_oracle': test_dual_oracle.cpu().numpy().tolist(),
                'test_lambda_mse': test_mse,
                'test_mean_pred_reward': test_mean_pred_reward,
                'test_mean_oracle_reward': test_mean_oracle_reward,
                'test_mean_regret': test_mean_regret,
                'algo': args.algo,
            }
            eval_backup.write_text(json.dumps(eval_data, indent=2))
        except Exception as e:
            print(f'[Evaluation] ERROR saving evaluation backup: {e}')
            import traceback
            traceback.print_exc()
    
    # For prof: magnitude of dℓ/dw = pred_loss (MSE); dv*/dw = 0 (v* does not depend on w). Weight balance uses ||∇_θ v*|| vs ||∇_θ pred_loss||.
    result = {
        'train_reward_per_epoch': train_reward_list,
        'train_lambda_pred': [l.cpu().numpy() for l in train_lambda_pred_list],
        'train_lambda_true': [l.cpu().numpy() for l in train_lambda_true_list],
        'test_obj_values': test_obj_values.cpu().numpy(),
        'test_oracle_values': test_oracle_values.cpu().numpy(),
        'test_regret': test_regret.cpu().numpy(),
        'test_lambda_mse': test_mse,
        'test_mean_pred_reward': test_mean_pred_reward,
        'test_mean_oracle_reward': test_mean_oracle_reward,
        'test_mean_regret': test_mean_regret,
        'test_lambda_pred': test_lambda_pred.cpu().numpy(),
        'test_lambda_true': test_lambda_true.cpu().numpy(),
        'obj_hat': obj_hat_list,
        'test_dual_hat': test_dual_hat,
        'test_dual_oracle': test_dual_oracle,
        'train_time': train_time_list,
        'sa_time_sec_per_epoch': sa_time_sec_per_epoch,
        'solver_time_sec_per_epoch': solver_time_sec_per_epoch,
        'train_mse_per_epoch': train_mse_list,  # MSE (prediction loss) for DFL; magnitude of dℓ/dw
        'train_dfl_loss_per_epoch': train_dfl_loss_list,  # Full DFL loss (-v*_θ + λ·pred_loss)
        'train_twostage_loss_per_epoch': train_twostage_loss_list,  # Prediction loss for two-stage
        'train_grad_norm_vstar_per_epoch': train_grad_norm_vstar_list,  # ||∇_θ v*|| (DFL only)
        'train_grad_norm_pred_loss_per_epoch': train_grad_norm_pred_loss_list,  # ||∇_θ pred_loss|| (DFL only)
        'train_zero_grad_frac_per_epoch': train_zero_grad_frac_list,  # fraction of |dV*/dλ| near zero (DFL only)
        'train_dv_dlambda_mean_per_epoch': train_dv_dlambda_mean_list,  # mean |dV*/dλ_pred| (DFL only)
        'train_order_bias_per_epoch': train_order_bias_per_epoch,
        'train_inv_bias_per_epoch': train_inv_bias_per_epoch,
        'train_lambda_bias_per_epoch': train_lambda_bias_per_epoch,
        'train_frac_over_order_per_epoch': train_frac_over_order_per_epoch,
        'train_optimal_reward': train_optimal_reward,  # mean optimal on train (for reward plot reference)
    }
    # Summary for loss-weight tuning: last-epoch gradient norms and MSE value
    if train_mse_list and train_grad_norm_vstar_list and train_grad_norm_pred_loss_list:
        result['mse_last_epoch'] = float(train_mse_list[-1])  # prediction loss value (MSE) at last epoch
        result['magnitude_grad_vstar_last_epoch'] = float(train_grad_norm_vstar_list[-1])  # ||∇_θ v*||
        result['magnitude_grad_pred_loss_last_epoch'] = float(train_grad_norm_pred_loss_list[-1])  # ||∇_θ ℓ_pred|| = magnitude of dℓ/dw
        # Backward compat (deprecated: was misnamed; use magnitude_grad_pred_loss for gradient, mse_last_epoch for loss value)
        result['magnitude_dl_dw_last_epoch'] = float(train_grad_norm_pred_loss_list[-1])
    return result





