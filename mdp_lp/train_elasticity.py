"""
Training and evaluation for inventory model with elasticity-based demand.

Implements DFL and two-stage methods with price-dependent Poisson demand.
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from mdp_lp.elasticity_model import ElasticityModel, LambdaPredictor
from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.utils import solve_lp
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver
from mdp_lp.soft_state_aggregation import SoftAggDense
from mdp_lp.solvers.qpth_qp_solver import solve_occupancy_qp_qpth


def prepare_elasticity_data(args):
    """Prepare data for elasticity-based inventory experiments."""
    device = torch.device("cpu")
    print(f"[Data Prep] Generating {args.n_samples} price scenarios...")
    print(f"[Data Prep] Experiment type: {args.experiment_type}")
    
    if args.experiment_type == 'a':
        p = torch.ones(args.n_samples, device=device) * args.p_constant
        p_related = None
    elif args.experiment_type == 'b':
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
        y, basis = solver_obj.solve(H, r, gamma)
        return y, basis
    
    obj_true_list = []
    for i in range(args.n_samples):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"[Data Prep] Solving MDP {i+1}/{args.n_samples}...")
        mdp = create_inventory_from_lambda(
            lambda_true[i:i+1],
            args.b_max, args.s_max, args.q_max, args.d_max,
            args.foc, args.voc, args.hc, args.bc, args.beta,
            p[i:i+1]
        )
        _, obj, _ = solve_lp(mdp, lp_solver)
        obj_true_list.append(obj[0])
    
    obj_true = torch.stack(obj_true_list)
    
    # Prepare features: [p, p_related (if applicable), noise]
    if p_related is not None:
        noise = torch.randn(args.n_samples) * 0.1
        feats = torch.stack([p, p_related, noise], dim=1)
        prices_tensor = torch.stack([p, p_related], dim=1)
    else:
        noise = torch.randn(args.n_samples) * 0.1
        feats = torch.stack([p, noise], dim=1)
        prices_tensor = p
    
    # Split into train/test
    train_size = int(0.8 * args.n_samples)
    test_size = args.n_samples - train_size
    train_dataset, test_dataset = random_split(
        TensorDataset(feats, obj_true, lambda_true, prices_tensor),
        [train_size, test_size]
    )
    
    if args.mode == 'train':
        train_tune_size = int(0.8 * train_size)
        val_size = train_size - train_tune_size
        train_dataset, test_dataset = random_split(train_dataset, [train_tune_size, val_size])
    
    # Use batching for speed (DFL remains correct; this only reduces Python/optimizer-step overhead).
    #
    # IMPORTANT for qpth/no-aggregation:
    #   The occupancy QP has SA variables where S=(b_max+s_max+1) and A=(q_max+1).
    #   qpth solves dense KKT systems whose cost/memory blows up quickly with SA, especially when batching.
    #   So we intentionally keep batch size very small for dfl_qpth when n_agg=-1.
    default_bs = 32
    if getattr(args, "algo", None) == "dfl_qpth" and getattr(args, "n_agg", -1) == -1:
        # Heuristic: for SA in the thousands (e.g., 151*21=3171), even B=2 can be painful on CPU.
        S = int(args.b_max + args.s_max + 1)
        A = int(args.q_max + 1)
        SA = int(S * A)
        default_bs = 1 if SA >= 1500 else 2
        print(f"[Data Prep] qpth no-agg detected (SA={SA}). Using small batch_size={default_bs} for feasibility/speed.")

    batch_size = min(default_bs, len(train_dataset)) if len(train_dataset) > 0 else 1
    test_batch_size = min(default_bs, len(test_dataset)) if len(test_dataset) > 0 else 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    
    mean_lambda = lambda_true[:train_size].mean().item()
    print(f"[Data Prep] Data preparation complete")
    
    return train_loader, test_loader, gt_elasticity, p, p_related, mean_lambda


def train_eval_elasticity(args, train_loader, test_loader, gt_elasticity, mean_lambda=None):
    """Train and evaluate with elasticity-based inventory model.
    
    Args:
        args: Training arguments
        train_loader: Training data loader
        test_loader: Test data loader
        gt_elasticity: Ground truth elasticity model
        mean_lambda: Mean lambda value from training data (for better initialization)
    """
    device = torch.device("cpu")
    hyperparam = args.hyperparam
    learning_rate = hyperparam['lr']
    hidden_dim = hyperparam['hidden_dim']
    if args.algo == 'twostage':
        hidden_dim = 128  # more capacity for two-stage only (ln(p) feats + log-space loss)
        learning_rate = 0.0005  # lower LR for two-stage: stabler convergence, better fit
    use_p_related = args.experiment_type == 'c'
    
    init_mean_lambda = mean_lambda if mean_lambda is not None else 1.0
    output_log = args.algo == 'twostage'  # log-space prediction only for two-stage (not DFL)
    pred_model = LambdaPredictor(hidden_dim, use_p_related=use_p_related, init_mean_lambda=init_mean_lambda, output_log=output_log).to(device)

    # Optional: learn a soft state aggregation mapping jointly with the predictor under DFL variants.
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
        y, basis = solver_obj.solve(H, r, gamma)
        return y, basis
    
    train_reward_list = []
    train_lambda_pred_list = []
    train_lambda_true_list = []
    obj_loss_list = []
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
    
    print(f"[Training] Starting: {args.n_epochs} epochs, algo={args.algo}")
    start_time = time.time()

    # qpth solver (only used for dfl_qpth)
    qp_solver = None
    if args.algo == "dfl_qpth":
        try:
            from qpth.qp import QPFunction, QPSolvers
            qp_solver = QPFunction(solver=QPSolvers.PDIPM_BATCHED)
        except Exception as e:
            raise RuntimeError(f"qpth is required for algo='dfl_qpth' but could not be imported/initialized: {e}")

    # Checkpointing can dominate runtime when we serialize huge arrays every epoch.
    # Default behavior remains: save every epoch, full partial results.
    checkpoint_every = int(getattr(args, "checkpoint_every", 1))
    save_full_partial = bool(getattr(args, "save_full_partial", True))
    if args.algo == "dfl_qpth" and getattr(args, "n_agg", -1) == -1:
        # qpth no-agg is already very expensive; keep I/O overhead minimal.
        checkpoint_every = max(checkpoint_every, 10)
        save_full_partial = False
    
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
        epoch_twostage_loss = []  # For two-stage: track prediction loss per batch
        epoch_grad_norm_vstar = []  # For DFL: ||∇_θ v*|| per batch
        epoch_grad_norm_pred_loss = []  # For DFL: ||∇_θ pred_loss|| per batch
        pred_model.train()
        
        for batch_data in train_loader:
            feats, obj_true, lambda_true, prices = batch_data
            batch_size = feats.shape[0]

            if args.algo == 'twostage':
                # Feed ln(p) [and ln(p_related)] so predictor matches DGP: ln(λ) = β₀ + β₁·ln(p) + ε
                feats_for_pred = feats.clone()
                feats_for_pred[:, 0] = torch.log(feats[:, 0] + 1e-10)
                if use_p_related and feats.shape[1] >= 3:
                    feats_for_pred[:, 1] = torch.log(feats[:, 1] + 1e-10)
                log_lambda_pred = pred_model(feats_for_pred)
                lambda_pred = torch.exp(log_lambda_pred)
            else:
                lambda_pred = pred_model(feats)
            if not torch.all(torch.isfinite(lambda_pred)):
                raise ValueError("lambda_pred contains non-finite values (NaN/Inf)")
            if torch.any(lambda_pred < 0):
                min_val = float(lambda_pred.min().item())
                raise ValueError(f"lambda_pred must be >= 0, but min(lambda_pred)={min_val}")

            if use_p_related:
                p = prices[:, 0]
            else:
                p = prices if prices.dim() == 1 else prices
            
            mdp = create_inventory_from_lambda(
                lambda_pred,
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
                if agg is not None:
                    P_use, r_use, gamma_use, _info = agg(
                        P_use, r_use, gamma_use, tau=getattr(args, "sa_tau", 0.5)
                    )
                    epoch_sa_time += float(_info.get("runtime_sec", 0.0))

                H_use = mdp.build_H(P_use)
                r_flat_use = r_use.reshape(batch_size, -1)

                if args.algo == "dfl":
                    # Differentiable LP via fixed-basis sensitivity (uses Gurobi to get a basis).
                    t0 = time.time()
                    y_star = DifferentiableLPSolver.apply(H_use, r_flat_use, gamma_use)
                    epoch_solver_time += (time.time() - t0)
                else:
                    # Differentiable QP (qpth) with quadratic regularization.
                    t0 = time.time()
                    y_star = solve_occupancy_qp_qpth(
                        H_use, r_flat_use, gamma_use, reg=getattr(args, "qp_reg", 0.5), qp_solver=qp_solver
                    )
                    epoch_solver_time += (time.time() - t0)

                v_star = torch.sum(r_flat_use * y_star, dim=1)  # (batch,)

                pred_loss = pred_loss_fn(lambda_pred, lambda_true)
                # Maximize (v* - loss_weight * pred_loss); minimize negative of that
                dfl_objective = v_star.mean() - args.loss_weight * pred_loss
                dfl_loss = -dfl_objective

                # For logging: compare to obj_true (computed from LP on lambda_true)
                obj_hat = v_star.detach()
                obj_diff = obj_diff_fn(obj_hat, obj_true)
                obj_loss += obj_diff.item()

                epoch_mse.append(pred_loss.item())
                epoch_dfl_loss.append(dfl_loss.item())

                reward = v_star.mean().item()
                epoch_rewards.append(reward)
                epoch_lambda_pred.append(lambda_pred.detach())
                epoch_lambda_true.append(lambda_true)

                # Gradient magnitudes for weight balance: ||∇_θ v*|| and ||∇_θ pred_loss|| (dℓ/dw = pred_loss)
                def _grad_norm(model):
                    total = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total += p.grad.data.pow(2).sum().item()
                    return float(np.sqrt(total)) if total > 0 else 0.0

                optimizer.zero_grad()
                (-v_star.mean()).backward(retain_graph=True)
                epoch_grad_norm_vstar.append(_grad_norm(pred_model))
                optimizer.zero_grad()
                pred_loss.backward(retain_graph=True)
                epoch_grad_norm_pred_loss.append(_grad_norm(pred_model))
                optimizer.zero_grad()
                dfl_loss.backward()
                train_loss += dfl_loss.item()

            else:
                # Two-stage: only prediction loss (no gradients through LP)
                # Gamma is already clamped in solve_lp() for non-degeneracy
                _, obj_hat, _ = solve_lp(mdp, lp_solver)
                obj_hat = obj_hat.detach()  # No gradients through LP
                # Log-space MSE for two-stage only (matches DGP: ln(λ) = β₀ + β₁ ln(p) + ε)
                log_lambda_true = torch.log(lambda_true + 1e-10)
                pred_loss = pred_loss_fn(log_lambda_pred, log_lambda_true)
                obj_diff = obj_diff_fn(obj_hat, obj_true)
                obj_loss += obj_diff.item()
                
                # Track prediction loss for plotting
                epoch_twostage_loss.append(pred_loss.item())
                
                # For two-stage, reward is the objective value (positive = better)
                reward = obj_hat.mean().item()
                epoch_rewards.append(reward)
                epoch_lambda_pred.append(lambda_pred.detach())
                epoch_lambda_true.append(lambda_true)
                
                loss = pred_loss
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
            
            optimizer.step()
        
        train_time_list.append(time.time() - start_time)
        sa_time_sec_per_epoch.append(epoch_sa_time)
        solver_time_sec_per_epoch.append(epoch_solver_time)
        avg_obj_loss = obj_loss / len(train_loader)
        obj_loss_list.append(avg_obj_loss)
        train_reward_list.append(np.mean(epoch_rewards))
        train_lambda_pred_list.append(torch.cat(epoch_lambda_pred))
        train_lambda_true_list.append(torch.cat(epoch_lambda_true))
        
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
        
        if (epoch + 1) % 10 == 0:
            print(f'[Training] Epoch {epoch + 1}/{args.n_epochs}, Loss: {train_loss / len(train_loader):.4f}')
        
        should_checkpoint = ((epoch + 1) % checkpoint_every == 0) or (epoch == 0) or ((epoch + 1) == args.n_epochs)
        if should_checkpoint and hasattr(args, 'save_dir') and args.save_dir:
            checkpoint_file = args.save_dir / 'checkpoint.json'
            model_file = args.save_dir / 'model_state.pt'
            results_partial_file = args.save_dir / 'results_partial.json'
            try:
                import json
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'train_reward_per_epoch': [float(r) for r in train_reward_list],
                    'obj_loss': [float(l) for l in obj_loss_list],
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
                    'train_reward_per_epoch': [float(r) for r in train_reward_list],
                    'obj_loss': [float(l) for l in obj_loss_list],
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
    
    print(f"[Training] Complete. Time: {time.time() - start_time:.2f}s")
    
    if hasattr(args, 'save_dir') and args.save_dir:
        try:
            final_checkpoint = args.save_dir / 'checkpoint_final.json'
            final_model = args.save_dir / 'model_state_final.pt'
            import json
            checkpoint_data = {
                'epoch': args.n_epochs,
                'train_reward_per_epoch': [float(r) for r in train_reward_list],
                'obj_loss': [float(l) for l in obj_loss_list],
                'train_time': train_time_list.copy(),
                'sa_time_sec_per_epoch': sa_time_sec_per_epoch.copy(),
                'solver_time_sec_per_epoch': solver_time_sec_per_epoch.copy(),
                'train_lambda_pred': [l.cpu().numpy().tolist() for l in train_lambda_pred_list],
                'train_lambda_true': [l.cpu().numpy().tolist() for l in train_lambda_true_list]
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
    test_obj_values = []
    test_lambda_pred = []
    test_lambda_true = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            feats, obj_true, lambda_true, prices = batch_data
            batch_size = feats.shape[0]

            if args.algo == 'twostage':
                feats_for_pred = feats.clone()
                feats_for_pred[:, 0] = torch.log(feats[:, 0] + 1e-10)
                if use_p_related and feats.shape[1] >= 3:
                    feats_for_pred[:, 1] = torch.log(feats[:, 1] + 1e-10)
                lambda_pred = torch.exp(pred_model(feats_for_pred))
            else:
                lambda_pred = pred_model(feats)
            if not torch.all(torch.isfinite(lambda_pred)):
                raise ValueError("lambda_pred contains non-finite values (NaN/Inf) during eval")
            if torch.any(lambda_pred < 0):
                min_val = float(lambda_pred.min().item())
                raise ValueError(f"lambda_pred must be >= 0, but min(lambda_pred)={min_val} during eval")

            if use_p_related:
                p = prices[:, 0]
            else:
                p = prices if prices.dim() == 1 else prices
            
            mdp = create_inventory_from_lambda(
                lambda_pred,
                args.b_max, args.s_max, args.q_max, args.d_max,
                args.foc, args.voc, args.hc, args.bc, args.beta,
                p
            )
            _, obj_hat, _ = solve_lp(mdp, lp_solver)
            test_obj_values.extend(obj_hat.tolist())
            
            test_lambda_pred.append(lambda_pred)
            test_lambda_true.append(lambda_true)
    
    test_obj_values = torch.tensor(test_obj_values)
    test_lambda_pred = torch.cat(test_lambda_pred)
    test_lambda_true = torch.cat(test_lambda_true)
    
    if hasattr(args, 'save_dir') and args.save_dir:
        try:
            eval_backup = args.save_dir / 'evaluation_backup.json'
            import json
            eval_data = {
                'test_obj_values': test_obj_values.cpu().numpy().tolist(),
                'test_lambda_pred': test_lambda_pred.cpu().numpy().tolist(),
                'test_lambda_true': test_lambda_true.cpu().numpy().tolist()
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
        'test_lambda_pred': test_lambda_pred.cpu().numpy(),
        'test_lambda_true': test_lambda_true.cpu().numpy(),
        'obj_loss': obj_loss_list,
        'train_time': train_time_list,
        'sa_time_sec_per_epoch': sa_time_sec_per_epoch,
        'solver_time_sec_per_epoch': solver_time_sec_per_epoch,
        'train_mse_per_epoch': train_mse_list,  # MSE (prediction loss) for DFL; magnitude of dℓ/dw
        'train_dfl_loss_per_epoch': train_dfl_loss_list,  # Full DFL loss (-v*_θ + λ·pred_loss)
        'train_twostage_loss_per_epoch': train_twostage_loss_list,  # Prediction loss for two-stage
        'train_grad_norm_vstar_per_epoch': train_grad_norm_vstar_list,  # ||∇_θ v*|| (DFL only)
        'train_grad_norm_pred_loss_per_epoch': train_grad_norm_pred_loss_list,  # ||∇_θ pred_loss|| (DFL only)
    }
    # Summary for loss-weight tuning: last-epoch gradient norms and MSE value
    if train_mse_list and train_grad_norm_vstar_list and train_grad_norm_pred_loss_list:
        result['mse_last_epoch'] = float(train_mse_list[-1])  # prediction loss value (MSE) at last epoch
        result['magnitude_grad_vstar_last_epoch'] = float(train_grad_norm_vstar_list[-1])  # ||∇_θ v*||
        result['magnitude_grad_pred_loss_last_epoch'] = float(train_grad_norm_pred_loss_list[-1])  # ||∇_θ ℓ_pred|| = magnitude of dℓ/dw
        # Backward compat (deprecated: was misnamed; use magnitude_grad_pred_loss for gradient, mse_last_epoch for loss value)
        result['magnitude_dl_dw_last_epoch'] = float(train_grad_norm_pred_loss_list[-1])
    return result





