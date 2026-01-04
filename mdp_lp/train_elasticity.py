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
    
    batch_size = int(0.8 * 0.2 * args.n_samples) if args.mode == 'train' else int(0.2 * args.n_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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
    use_p_related = args.experiment_type == 'c'
    
    init_mean_lambda = mean_lambda if mean_lambda is not None else 1.0
    pred_model = LambdaPredictor(hidden_dim, use_p_related=use_p_related, init_mean_lambda=init_mean_lambda).to(device)
    optimizer = optim.Adam(pred_model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()
    
    solver_obj = DenseLPSolver()
    def lp_solver(H, r, gamma):
        y, basis = solver_obj.solve(H, r, gamma)
        return y, basis
    
    train_reward_list = []
    train_lambda_pred_list = []
    train_lambda_true_list = []
    obj_loss_list = []
    train_time_list = []
    
    print(f"[Training] Starting: {args.n_epochs} epochs, algo={args.algo}")
    start_time = time.time()
    
    for epoch in range(args.n_epochs):
        train_loss = 0.0
        obj_loss = 0.0
        epoch_rewards = []
        epoch_lambda_pred = []
        epoch_lambda_true = []
        pred_model.train()
        
        for batch_data in train_loader:
            feats, obj_true, lambda_true, prices = batch_data
            batch_size = feats.shape[0]
            
            lambda_pred = pred_model(feats)
            
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
            
            if args.algo == 'dfl':
                H, r_flat, gamma = mdp.lp_formulation(mdp.P, mdp.r)
                y_star = DifferentiableLPSolver.apply(H, r_flat, gamma)
                obj_hat = torch.sum(r_flat * y_star, dim=1)
            else:
                _, obj_hat, _ = solve_lp(mdp, lp_solver)
                obj_hat = obj_hat.detach()
            
            pred_loss = loss_fn(lambda_pred, lambda_true)
            obj_diff = loss_fn(obj_hat, obj_true)
            obj_loss += obj_diff.item()
            
            reward = -obj_hat.mean().item()
            epoch_rewards.append(reward)
            epoch_lambda_pred.append(lambda_pred.detach())
            epoch_lambda_true.append(lambda_true)
            
            if args.algo == 'twostage':
                loss = pred_loss
            else:
                loss = obj_diff + args.loss_weight * pred_loss
            
            optimizer.zero_grad()
            if args.algo == 'dfl':
                H, r_flat, gamma = mdp.lp_formulation(mdp.P, mdp.r)
                y_star = DifferentiableLPSolver.apply(H, r_flat, gamma)
                obj_hat_grad = torch.sum(r_flat * y_star, dim=1)
                loss_grad = loss_fn(obj_hat_grad, obj_true) + args.loss_weight * pred_loss
                loss_grad.backward()
            else:
                loss.backward()
            
            optimizer.step()
            train_loss += loss.item()
        
        train_time_list.append(time.time() - start_time)
        avg_obj_loss = obj_loss / len(train_loader)
        obj_loss_list.append(avg_obj_loss)
        train_reward_list.append(np.mean(epoch_rewards))
        train_lambda_pred_list.append(torch.cat(epoch_lambda_pred))
        train_lambda_true_list.append(torch.cat(epoch_lambda_true))
        
        if (epoch + 1) % 10 == 0:
            print(f'[Training] Epoch {epoch + 1}/{args.n_epochs}, Loss: {train_loss / len(train_loader):.4f}')
        
        if hasattr(args, 'save_dir') and args.save_dir:
            checkpoint_file = args.save_dir / 'checkpoint.json'
            model_file = args.save_dir / 'model_state.pt'
            results_partial_file = args.save_dir / 'results_partial.json'
            try:
                import json
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'train_reward_per_epoch': [float(r) for r in train_reward_list],
                    'obj_loss': [float(l) for l in obj_loss_list],
                    'train_time': train_time_list.copy()
                }
                checkpoint_file.write_text(json.dumps(checkpoint_data, indent=2))
                torch.save(pred_model.state_dict(), model_file)
                
                partial_results = {
                    'epoch': epoch + 1,
                    'n_epochs': args.n_epochs,
                    'status': 'training_in_progress',
                    'train_reward_per_epoch': [float(r) for r in train_reward_list],
                    'train_lambda_pred': [l.cpu().numpy().tolist() for l in train_lambda_pred_list],
                    'train_lambda_true': [l.cpu().numpy().tolist() for l in train_lambda_true_list],
                    'obj_loss': [float(l) for l in obj_loss_list],
                    'train_time': train_time_list.copy(),
                    'test_obj_values': None,
                    'test_lambda_pred': None,
                    'test_lambda_true': None
                }
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
            
            lambda_pred = pred_model(feats)
            
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
    
    return {
        'train_reward_per_epoch': train_reward_list,
        'train_lambda_pred': [l.cpu().numpy() for l in train_lambda_pred_list],
        'train_lambda_true': [l.cpu().numpy() for l in train_lambda_true_list],
        'test_obj_values': test_obj_values.cpu().numpy(),
        'test_lambda_pred': test_lambda_pred.cpu().numpy(),
        'test_lambda_true': test_lambda_true.cpu().numpy(),
        'obj_loss': obj_loss_list,
        'train_time': train_time_list
    }





