#!/usr/bin/env python3
"""
Run all elasticity experiments with standard requirements:
- 100 epochs
- 100 samples (MDP cases)
- 100-200 states (151 states: b_max=50, s_max=100)
- 80/20 train/test split
"""

import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
import os


def run_experiment(exp_config):
    """Run a single elasticity experiment (100 epochs, 100 samples)."""
    experiment_type, cost_case, algo, beta_2 = exp_config
    
    cmd = [
        sys.executable, '-m', 'mdp_lp.run_elasticity_experiments',
        '--experiment_type', experiment_type,
        '--cost_case', cost_case,
        '--algo', algo,
        '--beta_2', str(beta_2),
        '--n_epochs', '100',
        '--n_samples', '100'
    ]
    
    # Create log file for this experiment
    log_dir = Path('mdp_lp/results_elasticity') / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if experiment_type == 'c':
        log_file = log_dir / f"exp_{experiment_type}_{cost_case}_{algo}_beta2_{beta_2}.log"
    else:
        log_file = log_dir / f"exp_{experiment_type}_{cost_case}_{algo}.log"
    
    print(f"[PID {os.getpid()}] Running: {experiment_type}, {cost_case}, {algo.upper()}, β₂={beta_2}")
    
    # Run with stdout/stderr redirected to log file
    with open(log_file, 'w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    
    if result.returncode == 0:
        print(f"[PID {os.getpid()}] Completed: {exp_config}")
        return True, exp_config, None
    else:
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                error_msg = ''.join(lines[-10:])[:500] if lines else "Unknown error"
        except:
            error_msg = "Could not read log file"
        print(f"[PID {os.getpid()}] Failed: {exp_config} - {error_msg}")
        return False, exp_config, error_msg


def main():
    # Run all experiments (DFL and Two-stage)
    experiments = [
        # DFL experiments
        ('a', 'high_margin', 'dfl', 0.0),
        ('a', 'low_margin', 'dfl', 0.0),
        ('b', 'high_margin', 'dfl', 0.0),
        ('b', 'low_margin', 'dfl', 0.0),
        ('c', 'high_margin', 'dfl', 0.5),   # β₂ > 0 (substitute)
        ('c', 'high_margin', 'dfl', -0.5),  # β₂ < 0 (complement)
        ('c', 'low_margin', 'dfl', 0.5),
        ('c', 'low_margin', 'dfl', -0.5),
        # Two-stage experiments
        ('a', 'high_margin', 'twostage', 0.0),
        ('a', 'low_margin', 'twostage', 0.0),
        ('b', 'high_margin', 'twostage', 0.0),
        ('b', 'low_margin', 'twostage', 0.0),
        ('c', 'high_margin', 'twostage', 0.5),
        ('c', 'high_margin', 'twostage', -0.5),
        ('c', 'low_margin', 'twostage', 0.5),
        ('c', 'low_margin', 'twostage', -0.5),
    ]
    
    
    num_cores = cpu_count()
    num_workers = max(1, min(num_cores - 2, len(experiments)))
    
    print("="*70)
    print("RUNNING ALL ELASTICITY EXPERIMENTS")
    print(f"Workers: {num_workers}/{num_cores} cores")
    print("Parameters: 100 epochs, 100 samples, 151 states")
    print("="*70)
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(run_experiment, experiments)
    
    success_count = sum(1 for success, _, _ in results if success)
    
    print("\n" + "="*70)
    print(f"EXPERIMENTS COMPLETE: {success_count}/{len(experiments)} successful")
    print("="*70)
    
    # Print failures
    failures = [(exp, err) for success, exp, err in results if not success]
    if failures:
        print("\nFAILED EXPERIMENTS:")
        for exp, err in failures:
            print(f"  {exp}: {err}")
    
    if success_count == len(experiments):
        print("\nGenerating comparison plots...")
        try:
            from mdp_lp.create_comparison_plots import create_all_comparison_plots
            create_all_comparison_plots()
            print("Comparison plots generated")
        except Exception as e:
            print(f"Warning: Could not generate comparison plots: {e}")
    
    try:
        from datetime import datetime
        completion_file = Path('mdp_lp/results_elasticity') / 'EXPERIMENTS_COMPLETE.txt'
        with open(completion_file, 'w') as f:
            f.write(f"EXPERIMENTS COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Success: {success_count}/{len(experiments)}\n")
            if failures:
                f.write(f"\nFailed experiments:\n")
                for exp, err in failures:
                    f.write(f"  {exp}: {err[:200]}\n")
            else:
                f.write("\nAll experiments completed successfully!\n")
            f.write(f"\nResults location: mdp_lp/results_elasticity/\n")
    except Exception as e:
        print(f"Warning: Could not create completion notification: {e}")


if __name__ == '__main__':
    main()
