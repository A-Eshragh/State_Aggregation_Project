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
from mdp_lp.utils import solve_lp
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver
from mdp_lp.solvers.differentiable_qp import DifferentiableQPSolver
from mdp_lp.soft_state_aggregation import SoftAggDense
from mdp_lp.solvers.qpth_qp_solver import solve_occupancy_qp_qpth
from mdp_lp.inventory_poisson_grad import poisson_pmf



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
            args.b_max, args.s_max, args.q_max, args.d_max,
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
        gen = torch.Generator().manual_seed(42)
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
    default_bs = 32
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
    """Pretrain the lambda prediction model."""
    learning_rate = 1e-2

    optimizer = optim.Adam(pred_model.parameters(), lr=learning_rate)
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

    agg = None
    params = list(pred_model.parameters())
    if getattr(args, "n_agg", -1) != -1 and args.algo in ("dfl", "dfl_qpth"):
        n_states = args.b_max + args.s_max + 1
        n_actions = args.q_max + 1
        agg = SoftAggDense(n_states=n_states, n_actions=n_actions, n_clusters=args.n_agg, r_embed=getattr(args, "sa_r_embed", 8)).to(device)
        params = params + list(agg.parameters())

    optimizer = optim.Adam(params, lr=learning_rate)
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

    qp_solver = QPFunction(eps=1e-10, maxIter=10, solver=QPSolvers.CVXPY)
    checkpoint_every = int(getattr(args, "checkpoint_every", 1))
    save_full_partial = bool(getattr(args, "save_full_partial", True))
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Zihao
    # if args.algo == "dfl_qpth" and getattr(args, "n_agg", -1) == -1:
    #     # qpth no-agg is already very expensive; keep I/O overhead minimal.
    #     checkpoint_every = max(checkpoint_every, 10)
    #     save_full_partial = False

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
                lambda_pred, args.b_max, args.s_max, args.q_max, args.d_max,
                args.foc, args.voc, args.hc, args.bc, args.beta, p
            )
            mdp_true = create_inventory_from_lambda(
                lambda_true, args.b_max, args.s_max, args.q_max, args.d_max,
                args.foc, args.voc, args.hc, args.bc, args.beta, p
            )
            S = mdp.P.shape[1]
            gamma = mdp.build_gamma(batch_size, S, device)
            gamma = torch.clamp(gamma, min=1e-8)
            gamma = gamma / gamma.sum(dim=1, keepdim=True)
            gamma_use = gamma
            if args.algo in ("dfl", "dfl_qpth"):
                P_use, r_use, gamma_use = mdp.P, mdp.r, gamma
                _agg_info = None
                if agg is not None:
                    P_use, r_use, gamma_use, _agg_info = agg(P_use, r_use, gamma_use, tau=getattr(args, "sa_tau", 0.5))
                H_use = mdp.build_H(P_use)
                r_flat_use = r_use.reshape(batch_size, -1)
                if args.algo == "dfl":
                    y_star = DifferentiableLPSolver.apply(H_use, r_flat_use, gamma_use)
                else:
                    y_star = solve_occupancy_qp_qpth(
                        H_use, r_flat_use, gamma_use,
                        reg=getattr(args, "qp_reg", 0.1), qp_solver=qp_solver
                    )
                if agg is not None and _agg_info is not None:
                    y_star = SoftAggDense.disaggregate_y(y_star, _agg_info["W"], mdp.r.shape[2])
            else:
                y_star, _, _, _ = solve_lp(mdp, lp_solver)
            r_true = mdp_true.r.reshape(batch_size, -1)
            v_star_init = (r_true * y_star).sum(dim=1).mean().item()
            initial_rewards.append(v_star_init)
    initial_reward = float(np.mean(initial_rewards))
    pred_model.train()

    if args.algo == "dfl" or args.algo == "dfl_qpth":
        pretrain_lambda_pred(pred_model, train_loader)

    for epoch in range(args.n_epochs):
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
        epoch_order_pred = []   # policy analysis: E[order] under predicted-λ policy
        epoch_order_oracle = [] # policy analysis: E[order] under oracle (true-λ) policy
        epoch_inv_pred = []     # policy analysis: E[inventory] under predicted-λ policy
        epoch_inv_oracle = []   # policy analysis: E[inventory] under oracle policy
        
        for batch_data in train_loader:
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
                args.b_max, args.s_max, args.q_max, args.d_max,
                args.foc, args.voc, args.hc, args.bc, args.beta,
                p
            )
            
            if args.algo in ("dfl", "dfl_qpth"):
                # Build gamma once (InventoryLP uses uniform gamma by default).
                S = mdp.P.shape[1]
                gamma = mdp.build_gamma(batch_size, S, device)
                gamma = torch.clamp(gamma, min=1e-8)
                gamma = gamma / gamma.sum(dim=1, keepdim=True)

                # Optionally aggregate (P, r, gamma) into M=n_agg clusters.
                P_use, r_use, gamma_use = mdp.P, mdp.r, gamma
                _agg_info = None
                if agg is not None:
                    P_use, r_use, gamma_use, _agg_info = agg(
                        P_use, r_use, gamma_use, tau=getattr(args, "sa_tau", 0.5)
                    )
                    epoch_sa_time += float(_agg_info.get("runtime_sec", 0.0))

                H_use = mdp.build_H(P_use)
                r_flat_use = r_use.reshape(batch_size, -1)

                if args.algo == "dfl":
                    t0 = time.time()
                    y_star = DifferentiableLPSolver.apply(H_use, r_flat_use, gamma_use)
                    epoch_solver_time += (time.time() - t0)
                elif args.algo == "dfl_qpth":
                    t0 = time.time()
                    y_star = solve_occupancy_qp_qpth(
                            H_use, r_flat_use, gamma_use,
                            reg=getattr(args, "qp_reg", 0.1),
                            qp_solver=qp_solver
                        )                    
                    epoch_solver_time += (time.time() - t0)
                else:
                    raise ValueError(f"Invalid algorithm: {args.algo}")

                # Disaggregate occupancy back to original state space for evaluation
                if agg is not None and _agg_info is not None:
                    y_star = SoftAggDense.disaggregate_y(y_star, _agg_info["W"], mdp.r.shape[2])

                with torch.no_grad():
                    mdp_true = create_inventory_from_lambda(lambda_true, args.b_max, args.s_max, args.q_max, args.d_max,
                                                            args.foc, args.voc, args.hc, args.bc, args.beta,
                                                            p)
                    r_true = mdp_true.r.reshape(batch_size, -1)
                    H_true = mdp_true.build_H(mdp_true.P)

                v_star, _y_pi_true = eval_y_on_true_mdp(mdp_true, y_star, gamma)
                dfl_loss = -v_star.mean() * 10000
                aug_lag = torch.tensor(0.0)

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
                print(f"lambda_pred: {lambda_pred.detach().cpu().numpy()}, lambda_true: {lambda_true.detach().cpu().numpy()}")

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
            mdp_true = create_inventory_from_lambda(lambda_true, args.b_max, args.s_max, args.q_max, args.d_max,
                                                    args.foc, args.voc, args.hc, args.bc, args.beta,
                                                    p)
            S_true = mdp_true.P.shape[1]
            gamma_true = mdp_true.build_gamma(batch_size, S_true, y_star.device)
            gamma_true = torch.clamp(gamma_true, min=1e-8)
            gamma_true = gamma_true / gamma_true.sum(dim=1, keepdim=True)
            v_eval, _ = eval_y_on_true_mdp(mdp_true, y_star, gamma_true)
            epoch_rewards.append(v_eval.mean().item())
        
        train_reward_list.append(float(np.mean(epoch_rewards)))
        train_time_list.append(time.time() - start_time)
        sa_time_sec_per_epoch.append(epoch_sa_time)
        solver_time_sec_per_epoch.append(epoch_solver_time)
        avg_obj_loss = obj_loss / len(train_loader)
        train_lambda_pred_list.append(torch.cat(epoch_lambda_pred))
        train_lambda_true_list.append(torch.cat(epoch_lambda_true))
        obj_hat_list.append(obj_hat)

        
        # For DFL variants (LP-based and qpth-based): track average MSE and full DFL loss per epoch
        if args.algo in ('dfl', 'dfl_qpth') and epoch_mse:
            train_mse_list.append(np.mean(epoch_mse))
            train_dfl_loss_list.append(np.mean(epoch_dfl_loss))
            train_twostage_loss_list.append(0.0)  # Placeholder for two-stage
            train_grad_norm_vstar_list.append(np.mean(epoch_grad_norm_vstar) if epoch_grad_norm_vstar else 0.0)
            train_grad_norm_pred_loss_list.append(np.mean(epoch_grad_norm_pred_loss) if epoch_grad_norm_pred_loss else 0.0)
        else:
            # Two-stage (or any other non-DFL mode): no DFL loss/MSE
            train_mse_list.append(0.0)
            train_dfl_loss_list.append(0.0)
            train_grad_norm_vstar_list.append(0.0)
            train_grad_norm_pred_loss_list.append(0.0)
            # For two-stage: track prediction loss
            if epoch_twostage_loss:
                train_twostage_loss_list.append(np.mean(epoch_twostage_loss))
            else:
                train_twostage_loss_list.append(0.0)
        
        # Policy analysis: aggregate per-epoch (or placeholders when --policy_analysis not set)
        _ep_lam_pred = torch.cat(epoch_lambda_pred)
        _ep_lam_true = torch.cat(epoch_lambda_true)
        train_lambda_bias_per_epoch.append(float((_ep_lam_pred - _ep_lam_true).mean()))
        if getattr(args, 'policy_analysis', False) and epoch_order_pred:
            _ep_ord_pred = torch.cat(epoch_order_pred)
            _ep_ord_oracle = torch.cat(epoch_order_oracle)
            _ep_inv_pred = torch.cat(epoch_inv_pred)
            _ep_inv_oracle = torch.cat(epoch_inv_oracle)
            _ep_ord_bias = _ep_ord_pred - _ep_ord_oracle
            train_order_bias_per_epoch.append(float(_ep_ord_bias.mean()))
            train_inv_bias_per_epoch.append(float((_ep_inv_pred - _ep_inv_oracle).mean()))
            train_frac_over_order_per_epoch.append(float((_ep_ord_bias > 0).float().mean()))
        else:
            train_order_bias_per_epoch.append(0.0)
            train_inv_bias_per_epoch.append(0.0)
            train_frac_over_order_per_epoch.append(0.0)
        
        if (epoch + 1) % 1 == 0:
            msg = f'[Training] Epoch {epoch + 1}/{args.n_epochs}, Loss: {train_loss / len(train_loader):.4f}, λ_bias={train_lambda_bias_per_epoch[-1]:+.3f}'
            if args.algo in ('dfl', 'dfl_qpth') and epoch_vstar and epoch_aug_lag and epoch_mse:
                msg += f'  |  v*={np.mean(epoch_vstar):.2f}  aug_lag={np.mean(epoch_aug_lag):.2f}  pred_MSE={np.mean(epoch_mse):.4f}'
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
            if torch.any(lambda_pred < 0):
                min_val = float(lambda_pred.min().item())
                print(f"[Evaluation] WARNING: clamping negative lambda_pred (min={min_val:.4f}) to 0.01")
                lambda_pred = torch.clamp(lambda_pred, min=0.01)

            if use_p_related:
                p = prices[:, 0]
            else:
                p = prices if prices.dim() == 1 else prices

            # --- Oracle: solve with true lambda ---
            mdp_true_test = create_inventory_from_lambda(
                lambda_true,
                args.b_max, args.s_max, args.q_max, args.d_max,
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
                lambda_pred,
                args.b_max, args.s_max, args.q_max, args.d_max,
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




