#!/usr/bin/env python3
"""
Run elasticity-based inventory experiments.

Experiments:
(a) β₂ = 0, p constant
(b) β₂ = 0, p from lognormal
(c) β₂ ≠ 0, p and p_related from lognormal (both positive and negative β₂)
"""

import argparse
import json
import sys
import numpy as np
import pathlib
import torch
from pathlib import Path
from mdp_lp.train_elasticity import prepare_elasticity_data, train_eval_elasticity
from mdp_lp.utils import set_seed
from mdp_lp.plot_elasticity import plot_elasticity_results


def main():
    print("[Experiment] Starting elasticity experiment")
    
    parser = argparse.ArgumentParser(description='Run elasticity-based inventory experiments')
    
    # Experiment type
    parser.add_argument('--experiment_type', type=str, choices=['a', 'b', 'c'], required=True,
                       help='Experiment type: a=β₂=0 p constant, b=β₂=0 p lognormal, c=β₂≠0')
    
    # Elasticity parameters (defaults fixed for consistency: λ = exp(β₀ + β₁·log(p) + β₂·log(p_related)))
    # ZIHAO
    parser.add_argument('--beta_0', type=float, default=4.0, help='β₀ in elasticity model')
    parser.add_argument('--beta_1', type=float, default=-0.3, help='β₁ in elasticity model')
    parser.add_argument('--beta_2', type=float, default=0.09, help='β₂ in elasticity model (exp a/b use 0; exp c use ±0.09)')
    parser.add_argument('--sigma_epsilon', type=float, default=0.7, help='σ_ε in elasticity model (higher = more λ variance; can be >1)')
    
    # Price parameters
    parser.add_argument('--p_constant', type=float, default=10.0, help='Constant price (experiment a)')
    parser.add_argument('--p_lognormal_mean', type=float, default=10.0, help='Mean of log(p) (experiments b, c)')
    parser.add_argument('--p_lognormal_std', type=float, default=1.0, help='Std of log(p) (higher = wider price spread; can be >1)')
    parser.add_argument('--p_related_lognormal_mean', type=float, default=8.0, help='Mean of log(p_related) (experiment c)')
    parser.add_argument('--p_related_lognormal_std', type=float, default=1.0, help='Std of log(p_related) (experiment c; can be >1)')
    
    # Inventory parameters
    parser.add_argument('--b_max', type=int, default=5, help='Maximum backlog')
    parser.add_argument('--s_max', type=int, default=50, help='Maximum storage')
    parser.add_argument('--q_max', type=int, default=50, help='Maximum order quantity')
    parser.add_argument('--d_max', type=int, default=50, help='Maximum demand')
    
    # Cost parameters (asymmetric cases)
    parser.add_argument('--cost_case', type=str, choices=['high_margin', 'low_margin', 'normal_margin'], required=True,
                       help='high_margin: hc=0.1; low_margin: hc=20; normal_margin: hc=2')
    parser.add_argument('--selling_price_base', type=float, default=10.0, help='Base selling price')
    # Optional cost overrides (useful for controlled hypothetical settings)
    parser.add_argument('--foc', type=float, default=None, help='Override fixed ordering cost (if set)')
    parser.add_argument('--voc', type=float, default=None, help='Override variable ordering cost (if set)')
    parser.add_argument('--hc', type=float, default=None, help='Override holding cost (if set)')
    parser.add_argument('--bc', type=float, default=None, help='Override backorder cost (if set)')
    # Optional tag to keep outputs separate without changing experiment_type
    parser.add_argument('--tag', type=str, default="", help='Optional tag appended to save_dir')
    
    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--n_samples', type=int, default=50, help='Number of samples')
    parser.add_argument('--train_frac', type=float, default=0.5, help='Fraction for train (rest = test). 0.5 = 50/50 split.')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available (MPS for Mac, CUDA otherwise)')
    parser.add_argument('--algo', type=str, choices=['dfl', 'twostage', 'dfl_qpth'], required=True, help='Algorithm')
    parser.add_argument('--loss_weight', type=float, default=20.0, help='Weight for prediction loss in DFL (balanced so v* and pred loss have similar gradient scale; no adaptation)')
    parser.add_argument('--hyperparam', type=str, default="{'lr': 0.005, 'hidden_dim': 32}",
                       help='Hyperparameters as dict string')
    parser.add_argument('--beta', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode')
    parser.add_argument('--results_root', type=str, default='mdp_lp/results_elasticity',
                        help='Root directory to save results (default: mdp_lp/results_elasticity)')

    # Soft state aggregation (optional; only used for DFL variants)
    parser.add_argument('--n_agg', type=int, default=-1,
                        help='Number of aggregated states/clusters (M). -1 disables aggregation.')
    parser.add_argument('--sa_tau', type=float, default=0.5, help='Soft aggregation temperature τ')
    parser.add_argument('--sa_r_embed', type=int, default=8, help='Soft aggregation destination embedding dim')

    # qpth QP regularization (only used for dfl_qpth)
    parser.add_argument('--qp_reg', type=float, default=1e-5, help='Quadratic regularization coef for QP-DFL')
    # DFL stability: gradient clipping and augmented Lagrangian
    parser.add_argument('--grad_clip', type=float, default=10.0, help='Max gradient norm for clipping (prevents explosion)')
    parser.add_argument('--aug_lag_rho', type=float, default=1.0, help='Augmented Lagrangian penalty coefficient ρ')
    parser.add_argument('--policy_analysis', action='store_true', help='If set, run policy analysis (1 extra LP per batch for order/inv bias vs oracle). Default: off (1 solve per batch).')
    
    args = parser.parse_args()
    
    import ast
    args.hyperparam = ast.literal_eval(args.hyperparam)
    
    # Default costs by case
    # ZIHAO
    default_costs = {
        "high_margin": dict(foc=0, voc=0.5, hc=0.1, bc=0),
        "low_margin": dict(foc=0, voc=0.5, hc=20.0, bc=0),
        "normal_margin": dict(foc=0, voc=0.5, hc=2.0, bc=0),
    }
    dc = default_costs[args.cost_case]
    foc_default, voc_default, hc_default, bc_default = dc["foc"], dc["voc"], dc["hc"], dc["bc"]

    # Apply overrides if provided; otherwise use defaults
    args.foc = foc_default if args.foc is None else float(args.foc)
    args.voc = voc_default if args.voc is None else float(args.voc)
    args.hc = hc_default if args.hc is None else float(args.hc)
    args.bc = bc_default if args.bc is None else float(args.bc)
    
    if args.experiment_type == 'a' or args.experiment_type == 'b':
        args.beta_2 = 0.0
    
    set_seed(42)
    
    train_loader, test_loader, gt_elasticity, p, p_related, mean_lambda = prepare_elasticity_data(args)
    
    results_root = Path(args.results_root)
    if args.experiment_type == 'c':
        save_dir = results_root / f"exp_{args.experiment_type}" / args.cost_case / args.algo / f"beta2_{args.beta_2}"
    else:
        save_dir = results_root / f"exp_{args.experiment_type}" / args.cost_case / args.algo
    if args.tag:
        save_dir = save_dir / f"tag_{args.tag}"

    # Add aggregation / QP-specific subfolders only when used to keep backward-compatible paths.
    if args.n_agg != -1 and args.algo in ('dfl', 'dfl_qpth'):
        save_dir = save_dir / f"n_agg_{args.n_agg}"
    if args.algo == 'dfl_qpth':
        save_dir = save_dir / f"qp_reg_{args.qp_reg}"

    save_dir = save_dir / f"epochs_{args.n_epochs}"
    save_dir.mkdir(parents=True, exist_ok=True)
    args.save_dir = save_dir
    
    results = train_eval_elasticity(args, train_loader, test_loader, gt_elasticity, mean_lambda=mean_lambda)
    
    result_file = save_dir / 'results.json'
    results_partial_file = save_dir / 'results_partial.json'
    
    # Convert numpy arrays and tensors to lists for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (Path, pathlib.Path)):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_to_json_serializable(results)
    args_dict = {k: convert_to_json_serializable(v) for k, v in vars(args).items()}
    
    backup_file = save_dir / 'results_backup.json'
    backup2_file = save_dir / 'results_backup2.json'
    
    try:
        full_data = {**args_dict, **results_serializable}
        full_data['status'] = 'complete'
        full_data['epoch'] = args.n_epochs
        result_file.write_text(json.dumps(full_data, indent=4))
        
        if not result_file.exists() or result_file.stat().st_size == 0:
            raise Exception("File was not written or is empty")
        
        if results_partial_file.exists():
            results_partial_file.write_text(json.dumps(full_data, indent=4))
        
        backup_file.write_text(json.dumps(full_data, indent=4))
        backup2_file.write_text(json.dumps(full_data, indent=4))
        
    except Exception as e:
        print(f"[Save] ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            result_file.write_text(json.dumps(results_serializable, indent=4))
            backup_file.write_text(json.dumps(results_serializable, indent=4))
        except Exception as e2:
            print(f"[Save] ERROR in fallback: {e2}")
            try:
                emergency_file = save_dir.parent / f'results_emergency_{save_dir.name}.json'
                emergency_file.write_text(json.dumps(results_serializable, indent=4))
            except Exception as e3:
                traceback.print_exc()
                raise Exception(f"Could not save results. Last error: {e3}")
    
    if result_file.exists() and result_file.stat().st_size > 0:
        try:
            verify_data = json.loads(result_file.read_text())
            required_keys = ['train_reward_per_epoch', 'test_obj_values', 'test_lambda_pred', 'test_lambda_true']
            missing = [k for k in required_keys if k not in verify_data]
            if missing:
                print(f"[Save] WARNING: Missing keys: {missing}")
        except Exception as e:
            print(f"[Save] WARNING: Could not verify: {e}")
    else:
        raise Exception("results.json was not saved properly")
    
    try:
        plot_elasticity_results(results, save_dir, args.algo)
    except Exception as e:
        print(f"[Plot] WARNING: Plotting failed: {e}")
    
    print(f"[Experiment] Complete")
    
    return results


if __name__ == '__main__':
    main()






