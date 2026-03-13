"""
Plotting functions for elasticity-based inventory experiments.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_elasticity_results(results, save_dir, algo):
    """Generate all required plots for elasticity experiments."""
    save_dir = Path(save_dir)
    
    # 1. Average total discounted reward per epoch (with optimal value line if available)
    optimal_reward = results.get('train_optimal_reward')
    plot_reward_per_epoch(results['train_reward_per_epoch'], save_dir, algo, optimal_reward=optimal_reward)
    
    # 2. For DFL: plot MSE (prediction loss) and full DFL loss (with regularization)
    if algo.startswith('dfl') and 'train_mse_per_epoch' in results and 'train_dfl_loss_per_epoch' in results:
        plot_dfl_mse(results['train_mse_per_epoch'], save_dir)
        plot_dfl_loss(results['train_dfl_loss_per_epoch'], save_dir)
    # For DFL: gradient magnitudes for weight balance (||∇_θ v*||, ||∇_θ pred_loss||)
    if algo.startswith('dfl') and 'train_grad_norm_vstar_per_epoch' in results and 'train_grad_norm_pred_loss_per_epoch' in results:
        plot_grad_norms(results['train_grad_norm_vstar_per_epoch'], results['train_grad_norm_pred_loss_per_epoch'], save_dir)
    
    # For two-stage: plot prediction loss
    if algo == 'twostage' and 'train_twostage_loss_per_epoch' in results:
        plot_twostage_loss(results['train_twostage_loss_per_epoch'], save_dir)
    
    # 3. Predicted vs true lambda histograms (final epoch)
    final_lambda_pred = results['train_lambda_pred'][-1]
    final_lambda_true = results['train_lambda_true'][-1]
    test_lambda_pred = results['test_lambda_pred']
    test_lambda_true = results['test_lambda_true']
    plot_lambda_histograms(final_lambda_pred, final_lambda_true, 
                          test_lambda_pred, test_lambda_true, save_dir, algo)
    
    # 4. Side-by-side box plot of final optimal value on test set
    plot_test_obj_boxplot(results['test_obj_values'], save_dir, algo)


def plot_reward_per_epoch(train_reward_per_epoch, save_dir, algo, optimal_reward=None):
    """Plot average total discounted reward per epoch. First point is epoch 0 (initial); then 1..n_epochs. If optimal_reward is set, draw a horizontal reference line. Consistent y scale."""
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    epochs = range(0, len(train_reward_per_epoch))
    ax.plot(epochs, train_reward_per_epoch, linewidth=2, color='blue' if algo.startswith('dfl') else 'red', label='Train reward')
    if optimal_reward is not None:
        ax.axhline(y=optimal_reward, color='green', linestyle='--', linewidth=1.5, label='Optimal (train)')
        ax.legend(loc='lower right', fontsize=8)
    all_y = list(train_reward_per_epoch) + ([optimal_reward] if optimal_reward is not None else [])
    if all_y:
        mn, mx = min(all_y), max(all_y)
        margin = max((mx - mn) * 0.05, 1e-9)
        ax.set_ylim(min(0, mn) - margin, mx + margin)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Avg Discounted Reward')
    ax.set_title(f'{algo.upper()} - Training Reward per Epoch')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_dir / 'train_reward_per_epoch.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_lambda_histograms(train_lambda_pred, train_lambda_true, 
                          test_lambda_pred, test_lambda_true, save_dir, algo):
    """Plot predicted vs true lambda histograms."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    def _plot_hist_or_vline(ax, arr, *, bins=30, alpha=0.5, label="", color=""):
        arr = np.asarray(arr, dtype=float).reshape(-1)
        if arr.size == 0:
            return
        # If distribution is (nearly) constant, a histogram can render as an
        # almost-invisible razor-thin bar. In that case, draw a thick vline.
        if float(np.std(arr)) < 1e-6:
            x = float(np.mean(arr))
            ax.axvline(x, color=color, linewidth=4.0, alpha=0.9, label=label)
            return
        ax.hist(arr, bins=bins, alpha=alpha, label=label, color=color)
    
    # Training set
    _plot_hist_or_vline(axes[0], train_lambda_true, bins=30, alpha=0.5, label='True', color='blue')
    _plot_hist_or_vline(axes[0], train_lambda_pred, bins=30, alpha=0.5, label='Predicted', color='red')
    axes[0].set_xlabel('λ (Poisson Rate)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{algo.upper()} - Training Set')
    axes[0].legend()
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Test set
    _plot_hist_or_vline(axes[1], test_lambda_true, bins=30, alpha=0.5, label='True', color='blue')
    _plot_hist_or_vline(axes[1], test_lambda_pred, bins=30, alpha=0.5, label='Predicted', color='red')
    axes[1].set_xlabel('λ (Poisson Rate)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{algo.upper()} - Test Set')
    axes[1].legend()
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    # Same x scale on both panels
    parts = [np.asarray(train_lambda_pred).reshape(-1), np.asarray(train_lambda_true).reshape(-1),
             np.asarray(test_lambda_pred).reshape(-1), np.asarray(test_lambda_true).reshape(-1)]
    all_lam = np.concatenate([p for p in parts if len(p) > 0]) if any(len(p) > 0 for p in parts) else np.array([])
    if len(all_lam):
        lo, hi = float(np.min(all_lam)), float(np.max(all_lam))
        margin = max((hi - lo) * 0.05, 1e-6)
        for ax in axes:
            ax.set_xlim(lo - margin, hi + margin)
    plt.tight_layout()
    plt.savefig(save_dir / 'lambda_histograms.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_test_obj_boxplot(test_obj_values, save_dir, algo):
    """Plot box plot: for each test sample, LP optimal value when MDP uses *predicted* λ (not true-MDP value). Consistent y scale."""
    test_obj_values = np.asarray(test_obj_values).reshape(-1)
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    ax.boxplot([test_obj_values], tick_labels=[algo.upper()])
    if len(test_obj_values):
        mn, mx = float(np.min(test_obj_values)), float(np.max(test_obj_values))
        margin = max((mx - mn) * 0.05, 1e-9)
        ax.set_ylim(mn - margin, mx + margin)
    ax.set_ylabel('LP optimal value')
    ax.set_title(f'{algo.upper()} - Test: LP optimal under predicted λ (one per sample)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_dir / 'test_obj_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_comparison_boxplot(dfl_values, twostage_values, save_dir, filename=None):
    """Plot side-by-side box plot comparing DFL and Two-stage."""
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    ax.boxplot([dfl_values, twostage_values], labels=['DFL', 'Two-stage'])
    ax.set_ylabel('Optimal Value')
    ax.set_title('Test Set Optimal Values Comparison')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    out = save_dir / (filename or 'comparison_boxplot.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()


def plot_dfl_mse(train_mse_per_epoch, save_dir):
    """Plot MSE (prediction loss) per epoch for DFL. Consistent y scale."""
    train_mse_per_epoch = np.asarray(train_mse_per_epoch)
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    epochs = range(1, len(train_mse_per_epoch) + 1)
    ax.plot(epochs, train_mse_per_epoch, linewidth=2, color='green')
    if len(train_mse_per_epoch):
        mn, mx = float(np.min(train_mse_per_epoch)), float(np.max(train_mse_per_epoch))
        margin = max((mx - mn) * 0.05, 1e-9)
        ax.set_ylim(max(0, mn - margin), mx + margin)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE (Prediction Loss)')
    ax.set_title('DFL - Prediction Loss (MSE) per Epoch')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_dir / 'train_mse_per_epoch.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_dfl_loss(train_dfl_loss_per_epoch, save_dir):
    """Plot full DFL loss (-v*_θ + λ·pred_loss) per epoch. Consistent y scale."""
    train_dfl_loss_per_epoch = np.asarray(train_dfl_loss_per_epoch)
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    epochs = range(1, len(train_dfl_loss_per_epoch) + 1)
    ax.plot(epochs, train_dfl_loss_per_epoch, linewidth=2, color='purple')
    if len(train_dfl_loss_per_epoch):
        mn, mx = float(np.min(train_dfl_loss_per_epoch)), float(np.max(train_dfl_loss_per_epoch))
        margin = max((mx - mn) * 0.05, 1e-9)
        ax.set_ylim(max(0, mn - margin), mx + margin)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('DFL Loss (-v*_θ + λ·MSE)')
    ax.set_title('DFL - Full Loss (with Regularization) per Epoch')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_dir / 'train_dfl_loss_per_epoch.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_grad_norms(train_grad_norm_vstar_per_epoch, train_grad_norm_pred_loss_per_epoch, save_dir):
    """Plot ||∇_θ v*|| and ||∇_θ pred_loss|| per epoch for weight balance (dℓ/dw = MSE). Consistent y scale."""
    vstar = np.asarray(train_grad_norm_vstar_per_epoch)
    pred = np.asarray(train_grad_norm_pred_loss_per_epoch)
    fig, ax = plt.subplots(figsize=(6.5, 2))
    epochs = range(1, len(vstar) + 1)
    ax.plot(epochs, vstar, linewidth=2, color='blue', label=r'$\|\nabla_\theta v^*\|$')
    ax.plot(epochs, pred, linewidth=2, color='green', label=r'$\|\nabla_\theta \ell_{\mathrm{pred}}\|$')
    if len(vstar) or len(pred):
        all_g = np.concatenate([vstar, pred])
        mn, mx = float(np.min(all_g)), float(np.max(all_g))
        margin = max((mx - mn) * 0.05, 1e-9)
        ax.set_ylim(max(0, mn - margin), mx + margin)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Gradient norm')
    ax.set_title('DFL - Gradient magnitudes (for weight balance)')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_dir / 'train_grad_norms_per_epoch.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_twostage_loss(train_twostage_loss_per_epoch, save_dir):
    """Plot prediction loss per epoch for two-stage. Consistent y scale."""
    train_twostage_loss_per_epoch = np.asarray(train_twostage_loss_per_epoch)
    fig, ax = plt.subplots(figsize=(6.5, 1.5))
    epochs = range(1, len(train_twostage_loss_per_epoch) + 1)
    ax.plot(epochs, train_twostage_loss_per_epoch, linewidth=2, color='orange')
    if len(train_twostage_loss_per_epoch):
        mn, mx = float(np.min(train_twostage_loss_per_epoch)), float(np.max(train_twostage_loss_per_epoch))
        margin = max((mx - mn) * 0.05, 1e-9)
        ax.set_ylim(max(0, mn - margin), mx + margin)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Prediction Loss (MSE)')
    ax.set_title('Two-stage - Prediction Loss per Epoch')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(save_dir / 'train_twostage_loss_per_epoch.png', dpi=300, bbox_inches='tight')
    plt.close()







