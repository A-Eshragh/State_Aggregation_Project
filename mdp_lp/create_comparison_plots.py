#!/usr/bin/env python3
"""
Create side-by-side comparison plots for DFL vs Two-stage methods.

This script should be run after all experiments complete to generate
the final comparison box plots as required.
"""

import json
import numpy as np
from pathlib import Path
from mdp_lp.plot_elasticity import plot_comparison_boxplot


def create_all_comparison_plots(epochs=100):
    """
    Create side-by-side box plots for all experiment configurations.
    
    Args:
        epochs: Number of epochs to use (default: 100). Results will be loaded from
                epochs_{epochs} subdirectory.
    """
    base_dir = Path('mdp_lp/results_elasticity')
    
    # All experiment configurations matching run_all_elasticity.py
    experiments = [
        ('a', 'high_margin', None),
        ('a', 'low_margin', None),
        ('b', 'high_margin', None),
        ('b', 'low_margin', None),
        ('c', 'high_margin', 0.5),
        ('c', 'high_margin', -0.5),
        ('c', 'low_margin', 0.5),
        ('c', 'low_margin', -0.5),
    ]
    
    for exp_type, cost_case, beta_2 in experiments:
        # Construct paths - check for epochs subdirectory first, fallback to old structure
        if exp_type == 'c':
            dfl_dir = base_dir / f"exp_{exp_type}" / cost_case / 'dfl' / f'beta2_{beta_2}'
            twostage_dir = base_dir / f"exp_{exp_type}" / cost_case / 'twostage' / f'beta2_{beta_2}'
            save_dir = base_dir / f"exp_{exp_type}" / cost_case / f'beta2_{beta_2}'
        else:
            dfl_dir = base_dir / f"exp_{exp_type}" / cost_case / 'dfl'
            twostage_dir = base_dir / f"exp_{exp_type}" / cost_case / 'twostage'
            save_dir = base_dir / f"exp_{exp_type}" / cost_case
        
        # Check for epochs subdirectory (new structure) or direct results.json (old structure)
        epochs_dir = f'epochs_{epochs}'
        dfl_results_file = dfl_dir / epochs_dir / 'results.json'
        twostage_results_file = twostage_dir / epochs_dir / 'results.json'
        
        # Fallback to old structure if epochs subdirectory doesn't exist
        if not dfl_results_file.exists():
            dfl_results_file = dfl_dir / 'results.json'
        if not twostage_results_file.exists():
            twostage_results_file = twostage_dir / 'results.json'
        
        if not dfl_results_file.exists():
            print(f"Missing DFL results: {dfl_results_file}")
            continue
        if not twostage_results_file.exists():
            print(f"Missing Two-stage results: {twostage_results_file}")
            continue
        
        try:
            with open(dfl_results_file) as f:
                dfl_results = json.load(f)
            with open(twostage_results_file) as f:
                twostage_results = json.load(f)
            
            dfl_values = np.array(dfl_results['test_obj_values'])
            twostage_values = np.array(twostage_results['test_obj_values'])
            
            # Create comparison plot
            save_dir.mkdir(parents=True, exist_ok=True)
            plot_comparison_boxplot(dfl_values, twostage_values, save_dir)
            
            exp_label = f"{exp_type}/{cost_case}"
            if beta_2 is not None:
                exp_label += f"/β₂={beta_2}"
            print(f"Created comparison plot: {exp_label}")
            
        except Exception as e:
            exp_label = f"{exp_type}/{cost_case}"
            if beta_2 is not None:
                exp_label += f"/β₂={beta_2}"
            print(f"Error creating comparison for {exp_label}: {e}")


if __name__ == '__main__':
    import sys
    # Default to 100 epochs, but allow override via command line
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f"Creating comparison plots for {epochs} epoch experiments...")
    create_all_comparison_plots(epochs=epochs)





