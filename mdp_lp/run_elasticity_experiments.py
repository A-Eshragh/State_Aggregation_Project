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
    
    # Elasticity parameters
    parser.add_argument('--beta_0', type=float, default=2.0, help='β₀ in elasticity model')
    parser.add_argument('--beta_1', type=float, default=-1.5, help='β₁ in elasticity model')
    parser.add_argument('--beta_2', type=float, default=0.0, help='β₂ in elasticity model')
    parser.add_argument('--sigma_epsilon', type=float, default=0.1, help='σ_ε in elasticity model')
    
    # Price parameters
    parser.add_argument('--p_constant', type=float, default=10.0, help='Constant price (experiment a)')
    parser.add_argument('--p_lognormal_mean', type=float, default=10.0, help='Mean of log(p) (experiments b, c)')
    parser.add_argument('--p_lognormal_std', type=float, default=0.5, help='Std of log(p) (experiments b, c)')
    parser.add_argument('--p_related_lognormal_mean', type=float, default=8.0, help='Mean of log(p_related) (experiment c)')
    parser.add_argument('--p_related_lognormal_std', type=float, default=0.5, help='Std of log(p_related) (experiment c)')
    
    # Inventory parameters
    parser.add_argument('--b_max', type=int, default=50, help='Maximum backlog')
    parser.add_argument('--s_max', type=int, default=100, help='Maximum storage')
    parser.add_argument('--q_max', type=int, default=20, help='Maximum order quantity')
    parser.add_argument('--d_max', type=int, default=30, help='Maximum demand')
    
    # Cost parameters (asymmetric cases)
    parser.add_argument('--cost_case', type=str, choices=['high_margin', 'low_margin'], required=True,
                       help='high_margin: (selling_price - ordering_cost) >> holding_cost, low_margin: vice versa')
    parser.add_argument('--selling_price_base', type=float, default=10.0, help='Base selling price')
    
    # Training parameters
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--n_samples', type=int, default=200, help='Number of samples')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available (MPS for Mac, CUDA otherwise)')
    parser.add_argument('--algo', type=str, choices=['dfl', 'twostage'], required=True, help='Algorithm')
    parser.add_argument('--loss_weight', type=float, default=0.5, help='Weight for prediction loss in DFL')
    parser.add_argument('--hyperparam', type=str, default="{'lr': 0.001, 'hidden_dim': 32}",
                       help='Hyperparameters as dict string')
    parser.add_argument('--beta', type=float, default=0.95, help='Discount factor')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Mode')
    
    args = parser.parse_args()
    
    import ast
    args.hyperparam = ast.literal_eval(args.hyperparam)
    
    if args.cost_case == 'high_margin':
        args.foc = 1.0
        args.voc = 0.5
        args.hc = 0.1
        args.bc = 0.2
    else:
        args.foc = 1.0
        args.voc = 0.5
        args.hc = 20.0
        args.bc = 0.2
    
    if args.experiment_type == 'a' or args.experiment_type == 'b':
        args.beta_2 = 0.0
    
    set_seed(42)
    
    train_loader, test_loader, gt_elasticity, p, p_related, mean_lambda = prepare_elasticity_data(args)
    
    if args.experiment_type == 'c':
        save_dir = Path('mdp_lp/results_elasticity') / f"exp_{args.experiment_type}" / args.cost_case / args.algo / f"beta2_{args.beta_2}" / f"epochs_{args.n_epochs}"
    else:
        save_dir = Path('mdp_lp/results_elasticity') / f"exp_{args.experiment_type}" / args.cost_case / args.algo / f"epochs_{args.n_epochs}"
    save_dir.mkdir(parents=True, exist_ok=True)
    args.save_dir = save_dir
    
    results = train_eval_elasticity(args, train_loader, test_loader, gt_elasticity, mean_lambda=mean_lambda)
    
    result_file = save_dir / 'results.json'
    results_partial_file = save_dir / 'results_partial.json'
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
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






