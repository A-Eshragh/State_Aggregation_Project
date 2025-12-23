#!/usr/bin/env python3
"""
Comprehensive experiment runner for LP-based MDP solving.
Runs experiments across multiple configurations for both BND and Inventory MDPs.
"""

import subprocess
import sys
from pathlib import Path


# Experiment configurations (matching QP results structure)
# Use fewer epochs/samples for faster testing - increase for final runs
BND_CONFIGS = [
    {
        'n_states': 2000,
        'n_actions': 10,
        'n_neighbors': 2,
        'two_way': True,
        'n_epochs': 30,  # Using epochs 20-30
        'n_samples': 50,  # Reduced from 100 for faster testing
    },
    {
        'n_states': 5000,
        'n_actions': 10,
        'n_neighbors': 1,
        'two_way': False,
        'n_epochs': 30,
        'n_samples': 50,
    },
    {
        'n_states': 10000,
        'n_actions': 10,
        'n_neighbors': 5,
        'two_way': True,
        'n_epochs': 30,
        'n_samples': 50,
    },
]

INVENTORY_CONFIGS = [
    {
        'b_max': 300,
        's_max': 500,
        'q_max': 10,
        'd_max': 8,
        'n_epochs': 30,
        'n_samples': 50,
    },
]

# Hyperparameter configurations (same as main repo)
HYPERPARAMS = [
    {'lr': 5e-3, 'hidden_dim': 32},
    {'lr': 5e-3, 'hidden_dim': 64},
    {'lr': 1e-3, 'hidden_dim': 32},
    {'lr': 1e-3, 'hidden_dim': 64},
    {'lr': 5e-4, 'hidden_dim': 32},
    {'lr': 5e-4, 'hidden_dim': 64},
    {'lr': 1e-4, 'hidden_dim': 32},
    {'lr': 1e-4, 'hidden_dim': 64},
]

ALGORITHMS = ['dfl', 'twostage']


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed with return code {result.returncode}")
        return False
    return True


def run_training(mdp_type, config, algo, hp_idx, hp):
    """Run training for a specific configuration."""
    # Convert dict to string format that argparse can parse (ast.literal_eval format)
    hp_str = str(hp)  # e.g., "{'lr': 0.001, 'hidden_dim': 32}"
    
    cmd = [
        sys.executable, '-m', 'mdp_lp.main', mdp_type,
        '--algo', algo,
        '--mode', 'train',
        '--n_epochs', str(config['n_epochs']),
        '--n_samples', str(config['n_samples']),
        '--hyperparam', hp_str,
    ]
    
    if mdp_type == 'bnd':
        cmd.extend([
            '--n_states', str(config['n_states']),
            '--n_actions', str(config['n_actions']),
            '--n_neighbors', str(config['n_neighbors']),
            '--two_way', 'y' if config['two_way'] else 'n',
        ])
    else:  # inventory
        cmd.extend([
            '--b_max', str(config['b_max']),
            '--s_max', str(config['s_max']),
            '--q_max', str(config['q_max']),
            '--d_max', str(config['d_max']),
        ])
    
    description = f"{mdp_type.upper()} - {algo.upper()} - Config {hp_idx+1}/{len(HYPERPARAMS)}"
    return run_command(cmd, description)


def run_test(mdp_type, config, algo):
    """Run test mode (uses best hyperparameters from training)."""
    cmd = [
        sys.executable, '-m', 'mdp_lp.main', mdp_type,
        '--algo', algo,
        '--mode', 'test',
    ]
    
    if mdp_type == 'bnd':
        cmd.extend([
            '--n_states', str(config['n_states']),
            '--n_actions', str(config['n_actions']),
            '--n_neighbors', str(config['n_neighbors']),
            '--two_way', 'y' if config['two_way'] else 'n',
        ])
    else:  # inventory
        cmd.extend([
            '--b_max', str(config['b_max']),
            '--s_max', str(config['s_max']),
            '--q_max', str(config['q_max']),
            '--d_max', str(config['d_max']),
        ])
    
    description = f"{mdp_type.upper()} - {algo.upper()} - TEST MODE"
    return run_command(cmd, description)


def run_plotting(mdp_type, config):
    """Generate plots for a configuration."""
    cmd = [
        sys.executable, '-m', 'mdp_lp.plot_results',
        '--mdp_type', mdp_type,
    ]
    
    if mdp_type == 'bnd':
        cmd.extend([
            '--n_states', str(config['n_states']),
            '--n_actions', str(config['n_actions']),
            '--n_neighbors', str(config['n_neighbors']),
            '--two_way', 'y' if config['two_way'] else 'n',
        ])
    else:  # inventory
        # Plotting for inventory might need different args
        print(f"Skipping plotting for inventory (not implemented yet)")
        return True
    
    description = f"{mdp_type.upper()} - PLOTTING"
    return run_command(cmd, description)


def main():
    """Run all experiments."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run LP MDP experiments')
    parser.add_argument('--mdp_type', type=str, choices=['bnd', 'inventory', 'all'], 
                       default='all', help='MDP type to run')
    parser.add_argument('--skip_train', action='store_true', 
                       help='Skip training (assume already done)')
    parser.add_argument('--skip_test', action='store_true',
                       help='Skip test mode')
    parser.add_argument('--skip_plot', action='store_true',
                       help='Skip plotting')
    parser.add_argument('--algo', type=str, choices=['dfl', 'twostage', 'all'],
                       default='all', help='Algorithm to run')
    parser.add_argument('--config_idx', type=int, default=None,
                       help='Run only specific config index (for parallelization)')
    parser.add_argument('--n_epochs', type=int, default=None,
                       help='Override n_epochs for all configs (for faster testing, max 20 recommended)')
    parser.add_argument('--n_samples', type=int, default=None,
                       help='Override n_samples for all configs (for faster testing)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LP MDP EXPERIMENT RUNNER")
    print("="*70)
    print("This will run comprehensive experiments similar to QP results")
    print("="*70)
    
    # Determine which MDPs to run
    mdp_types = ['bnd', 'inventory'] if args.mdp_type == 'all' else [args.mdp_type]
    algos = ALGORITHMS if args.algo == 'all' else [args.algo]
    
    total_experiments = 0
    experiment_count = 0
    
    for mdp_type in mdp_types:
        configs = BND_CONFIGS if mdp_type == 'bnd' else INVENTORY_CONFIGS
        
        for config_idx, config in enumerate(configs):
            # Skip if specific config requested
            if args.config_idx is not None and config_idx != args.config_idx:
                continue
            
            # Override epochs/samples if provided
            if args.n_epochs is not None:
                config = config.copy()
                config['n_epochs'] = args.n_epochs
            if args.n_samples is not None:
                config = config.copy()
                config['n_samples'] = args.n_samples
                
            for algo in algos:
                # Training phase - hyperparameter tuning
                if not args.skip_train:
                    print(f"\n{'#'*70}")
                    print(f"TRAINING PHASE: {mdp_type.upper()} - {algo.upper()} - Config {config_idx+1}")
                    print(f"  Config: {config}")
                    print(f"{'#'*70}\n")
                    
                    for hp_idx, hp in enumerate(HYPERPARAMS):
                        success = run_training(mdp_type, config, algo, hp_idx, hp)
                        if not success:
                            print(f"WARNING: Training failed for {mdp_type} {algo} hp={hp}")
                        total_experiments += 1
                        experiment_count += 1
                        print(f"\n✓ Completed {experiment_count} experiments\n")
                
                # Test phase - uses best hyperparameters
                if not args.skip_test:
                    print(f"\n{'#'*70}")
                    print(f"TEST PHASE: {mdp_type.upper()} - {algo.upper()} - Config {config_idx+1}")
                    print(f"{'#'*70}\n")
                    
                    run_test(mdp_type, config, algo)
            
            # Plotting phase (once per config, after both algorithms)
            if not args.skip_plot:
                print(f"\n{'#'*70}")
                print(f"PLOTTING PHASE: {mdp_type.upper()} - Config {config_idx+1}")
                print(f"{'#'*70}\n")
                
                run_plotting(mdp_type, config)
    
    print("\n" + "="*70)
    print(f"EXPERIMENTS COMPLETE!")
    print(f"Total training runs: {total_experiments}")
    print(f"Results saved in: mdp_lp/results_lp/")
    print("="*70)


if __name__ == '__main__':
    main()
