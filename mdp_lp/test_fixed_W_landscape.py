"""
Plot-only: loss landscape L(λ) for a saved fixed aggregation matrix W.

This script is meant to be run on an experiment output directory like:

  mdp_lp/results_elasticity/exp_b/high_margin/dfl/n_agg_10/epochs_300/

where we expect:
  - results.json (for MDP parameters)
  - agg_W_fixed.pt (saved fixed W)   OR you can pass --w_path explicitly

We sweep λ over a grid, build the (original) MDP at λ, aggregate (P,r,γ) with the
saved fixed W, solve the aggregated occupancy LP with SciPy HiGHS, disaggregate
occupancy back to original state space, and compute:

    L(λ) = - r_true(λ_ref)^T y_disagg(λ) + ρ * mean((γ_ref - H_ref y_disagg(λ))^2)

If --lambda_ref / --price_ref are omitted, the script reconstructs the saved
experiment sample and uses that sample's true lambda and price by default.
The scan also applies the same lambda floor used during training.
"""

import time
import json
import argparse
from pathlib import Path
from typing import Optional
import torch
import numpy as np

from mdp_lp.elasticity_model import ElasticityModel
from mdp_lp.inventory_elasticity import create_inventory_from_lambda
from mdp_lp.soft_state_aggregation import SoftAggDense
from mdp_lp.utils import set_seed

try:
    from scipy.optimize import linprog  # type: ignore
except Exception as e:
    raise RuntimeError(
        "This script requires SciPy (for linprog/HiGHS). Install with `pip install scipy`."
    ) from e


def solve_lp_scipy(H: torch.Tensor, r: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    """
    Forward-only LP solve using SciPy HiGHS:
        maximize r^T y  s.t. H y = gamma, y >= 0
    Inputs are single-sample tensors:
        H: (S, SA), r: (SA,), gamma: (S,)
    Returns:
        y: (SA,) torch tensor (float32)
    """
    H_np = H.detach().cpu().numpy()
    r_np = r.detach().cpu().numpy()
    g_np = gamma.detach().cpu().numpy()

    # linprog minimizes; convert max r^T y -> min (-r)^T y
    c = -r_np
    res = linprog(
        c=c,
        A_eq=H_np,
        b_eq=g_np,
        bounds=(0.0, None),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"SciPy linprog failed: status={res.status} msg={res.message}")
    y = torch.tensor(res.x, dtype=torch.float32)
    return y


def _aggregate_with_fixed_W(P: torch.Tensor, r: torch.Tensor, gamma: torch.Tensor, W: torch.Tensor, eps: float = 1e-8):
    """
    Aggregate (P, r, gamma) with a fixed membership matrix W.

    Args:
        P: (B,S,A,S)
        r: (B,S,A)
        gamma: (B,S)
        W: (S,M) or (B,S,M)
    Returns:
        P_agg: (B,M,A,M)
        r_agg: (B,M,A)
        gamma_agg: (B,M)
        info: dict with 'W' (B,S,M)
    """
    if W.dim() == 2:
        Wb = W.unsqueeze(0).expand(P.shape[0], -1, -1)
    else:
        Wb = W
    mass = Wb.sum(dim=1).clamp_min(eps)  # (B,M)
    r_agg = torch.einsum("bsm,bsa->bma", Wb, r) / mass.unsqueeze(-1)
    right = torch.einsum("bsaj,bjm->bsam", P, Wb)
    p_hat = torch.einsum("bsm,bsan->bman", Wb, right)
    P_agg = p_hat / mass[:, :, None, None]
    P_agg = P_agg.clamp_min(0.0)
    P_agg = P_agg / P_agg.sum(dim=-1, keepdim=True).clamp_min(eps)
    gamma_agg = torch.einsum("bs,bsm->bm", gamma, Wb).clamp_min(eps)
    gamma_agg = gamma_agg / gamma_agg.sum(dim=1, keepdim=True).clamp_min(eps)
    return P_agg, r_agg, gamma_agg, {"W": Wb}


def _uniform_gamma(mdp):
    S = mdp.P.shape[1]
    g = mdp.build_gamma(1, S, torch.device("cpu"))
    g = torch.clamp(g, min=1e-8)
    return g / g.sum(dim=1, keepdim=True)


def _sample_prices_and_lambdas_from_cfg(args_cfg: dict):
    """Recreate the experiment's sampled prices and lambda_true values."""
    set_seed(int(args_cfg["seed"]))
    device = torch.device("cpu")
    n_samples = int(args_cfg.get("generated_n_samples", args_cfg["n_samples"]))
    experiment_type = str(args_cfg["experiment_type"])

    if experiment_type == "a":
        p = torch.ones(n_samples, device=device) * float(args_cfg["p_constant"])
        p_related = None
    elif experiment_type == "b":
        if bool(args_cfg.get("fixed_p", False)):
            p = torch.arange(1, n_samples + 1, dtype=torch.float32, device=device)
        else:
            p = torch.exp(
                torch.randn(n_samples, device=device) * float(args_cfg["p_lognormal_std"])
                + np.log(float(args_cfg["p_lognormal_mean"]))
            )
        p_related = None
    elif experiment_type == "c":
        p = torch.exp(
            torch.randn(n_samples, device=device) * float(args_cfg["p_lognormal_std"])
            + np.log(float(args_cfg["p_lognormal_mean"]))
        )
        p_related = torch.exp(
            torch.randn(n_samples, device=device) * float(args_cfg["p_related_lognormal_std"])
            + np.log(float(args_cfg["p_related_lognormal_mean"]))
        )
    else:
        raise ValueError(f"Unsupported experiment_type={experiment_type!r}")

    gt_elasticity = ElasticityModel(
        beta_0=float(args_cfg["beta_0"]),
        beta_1=float(args_cfg["beta_1"]),
        beta_2=float(args_cfg["beta_2"]),
        sigma_epsilon=float(args_cfg["sigma_epsilon"]),
    )
    lambda_true = gt_elasticity(p, p_related)
    return p.cpu(), None if p_related is None else p_related.cpu(), lambda_true.cpu()


def _split_dataset_indices(n_samples: int, seed: int):
    if n_samples <= 1:
        return [0], [0]
    train_size = n_samples // 2
    perm = torch.randperm(n_samples, generator=torch.Generator().manual_seed(int(seed))).tolist()
    return perm[:train_size], perm[train_size:]


def _resolve_reference_values(
    args_cfg: dict,
    ref_split: str,
    ref_index: int,
    lambda_ref_override: Optional[float],
    price_ref_override: Optional[float],
):
    if lambda_ref_override is not None and price_ref_override is not None:
        return float(lambda_ref_override), float(price_ref_override), {
            "source": "explicit",
            "split": ref_split,
            "ref_index": ref_index,
            "sample_index": None,
            "p_related_ref": None,
        }

    p_saved = args_cfg.get("p_values_used")
    p_related_saved = args_cfg.get("p_related_values_used")
    p, p_related, lambda_true = _sample_prices_and_lambdas_from_cfg(args_cfg)
    if p_saved is not None:
        p = torch.as_tensor(p_saved, dtype=torch.float32)
    if p_related_saved is not None:
        p_related = torch.as_tensor(p_related_saved, dtype=torch.float32)

    n_samples = int(p.shape[0])
    train_idx, test_idx = _split_dataset_indices(n_samples, int(args_cfg["seed"]))
    if ref_split == "train":
        split_indices = train_idx
    elif ref_split == "test":
        split_indices = test_idx
    elif ref_split == "all":
        split_indices = list(range(n_samples))
    else:
        raise ValueError(f"Unsupported reference split {ref_split!r}")

    if not split_indices:
        raise ValueError(f"No samples available in split {ref_split!r}")
    if ref_index < 0 or ref_index >= len(split_indices):
        raise IndexError(
            f"reference_index={ref_index} is out of range for split {ref_split!r} "
            f"(size={len(split_indices)})"
        )

    sample_index = split_indices[ref_index]
    lambda_ref = float(lambda_true[sample_index].item()) if lambda_ref_override is None else float(lambda_ref_override)
    price_ref = float(p[sample_index].item()) if price_ref_override is None else float(price_ref_override)
    p_related_ref = None if p_related is None else float(p_related[sample_index].item())
    return lambda_ref, price_ref, {
        "source": "reconstructed" if p_saved is None else "saved_prices+reconstructed_lambda",
        "split": ref_split,
        "ref_index": ref_index,
        "sample_index": sample_index,
        "p_related_ref": p_related_ref,
    }


def loss_landscape_forward(
    lam_val: float,
    W_fixed: torch.Tensor,
    args_cfg: dict,
    lambda_ref: float,
    price_ref: float,
    lambda_floor: float,
) -> float:
    """Forward-only L(λ) using SciPy linprog and a saved fixed W."""
    with torch.no_grad():
        b_max, s_max, q_max = int(args_cfg["b_max"]), int(args_cfg["s_max"]), int(args_cfg["q_max"])
        foc, voc, hc, bc = float(args_cfg["foc"]), float(args_cfg["voc"]), float(args_cfg["hc"]), float(args_cfg["bc"])
        beta = float(args_cfg["beta"])
        rho = float(args_cfg.get("aug_lag_rho", 10000.0))
        normalize_disagg = bool(args_cfg.get("normalize_disagg", True))

        # Match training: lambda_pred is clamped below by 1.0 before building the MDP.
        lam_eff = max(float(lam_val), float(lambda_floor))
        lam = torch.as_tensor([lam_eff], dtype=torch.float32)
        p = torch.as_tensor([float(price_ref)], dtype=torch.float32)
        mdp = create_inventory_from_lambda(lam, b_max, s_max, q_max, foc, voc, hc, bc, beta, p)
        gamma = _uniform_gamma(mdp)
        P_agg, r_agg, gamma_agg, info = _aggregate_with_fixed_W(mdp.P, mdp.r, gamma, W_fixed)

        # Build H for aggregated MDP and solve occupancy
        H_agg = mdp.build_H(P_agg)[0]  # (M, M*A) for B=1
        r_flat_agg = r_agg.reshape(1, -1)[0]
        gamma_agg = gamma_agg[0]
        y_agg = solve_lp_scipy(H_agg, r_flat_agg, gamma_agg).view(1, -1)

        # Disaggregate occupancy for loss computation
        n_actions = q_max + 1
        y_orig = SoftAggDense.disaggregate_y(y_agg, info["W"], n_actions)
        if normalize_disagg:
            expected_sum = 1.0 / (1.0 - beta)
            y_sum = y_orig.sum(dim=1, keepdim=True).clamp_min(1e-8)
            y_orig = y_orig * (expected_sum / y_sum)

        # Reference quantities at (lambda_ref, price_ref)
        lam_ref = torch.as_tensor([float(lambda_ref)], dtype=torch.float32)
        mdp_ref = create_inventory_from_lambda(lam_ref, b_max, s_max, q_max, foc, voc, hc, bc, beta, p)
        r_ref_flat = mdp_ref.r.reshape(1, -1)
        H_ref = mdp_ref.build_H(mdp_ref.P)
        gamma_ref = _uniform_gamma(mdp_ref)

        dfl = -(r_ref_flat * y_orig).sum().item()
        violation = gamma_ref - (H_ref @ y_orig.unsqueeze(-1)).squeeze(-1)
        pen = violation.pow(2).mean().item()
        return float(dfl + rho * pen)


def main():
    parser = argparse.ArgumentParser(description="Plot L(lambda) landscape using saved fixed W.")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="",
        help="Experiment output directory containing results.json and agg_W_fixed.pt",
    )
    parser.add_argument(
        "--w_path",
        type=str,
        default="",
        help="Optional explicit path to agg_W_fixed.pt (overrides results_dir/agg_W_fixed.pt).",
    )
    parser.add_argument(
        "--lambda_ref",
        type=float,
        default=None,
        help="Optional explicit reference lambda. If omitted, infer from the saved experiment sample.",
    )
    parser.add_argument(
        "--price_ref",
        type=float,
        default=None,
        help="Optional explicit reference price. If omitted, infer from the saved experiment sample.",
    )
    parser.add_argument(
        "--reference_split",
        type=str,
        choices=["train", "test", "all"],
        default="train",
        help="Which saved sample split to use when inferring lambda_ref/price_ref.",
    )
    parser.add_argument(
        "--reference_index",
        type=int,
        default=0,
        help="Index inside the selected reference split when inferring lambda_ref/price_ref.",
    )
    parser.add_argument("--lam_min", type=float, default=5.0)
    parser.add_argument("--lam_max", type=float, default=6.0)
    parser.add_argument("--n_grid", type=int, default=1000)
    parser.add_argument(
        "--lambda_floor",
        type=float,
        default=1.0,
        help="Clamp scan lambda to at least this value to match training-time lambda_pred flooring.",
    )
    parser.add_argument("--out_dir", type=str, default="mdp_lp/plot_results")
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    elif args.w_path:
        results_dir = Path(args.w_path).resolve().parent
    else:
        raise ValueError("Provide --results_dir or --w_path.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((results_dir / "results.json").read_text())
    w_path = Path(args.w_path) if args.w_path else (results_dir / "agg_W_fixed.pt")
    payload = torch.load(w_path, map_location="cpu")
    if isinstance(payload, dict) and "W_fixed" in payload:
        W_fixed = payload["W_fixed"].detach().cpu()
        fixed_seed = int(payload.get("fixed_W_seed", cfg.get("agg_fixed_W_seed", -1)))
        w_n_states = payload.get("n_states", None)
        w_n_actions = payload.get("n_actions", None)
        w_n_clusters = payload.get("n_clusters", None)
    else:
        raise RuntimeError(f"Expected a dict with key 'W_fixed' in {w_path}")

    # Sanity checks
    if W_fixed.dim() != 2:
        raise ValueError(f"W_fixed must be 2D (S,M); got {tuple(W_fixed.shape)}")
    # Ensure the saved W matches the experiment config.
    cfg_seed = cfg.get("agg_fixed_W_seed", None)
    cfg_n_agg = cfg.get("n_agg", None)
    cfg_n_states = int(cfg["b_max"]) + int(cfg["s_max"]) + 1
    cfg_n_actions = int(cfg["q_max"]) + 1

    if cfg_seed is not None and int(cfg_seed) != int(fixed_seed):
        raise RuntimeError(
            f"Seed mismatch: results.json has agg_fixed_W_seed={cfg_seed} but {w_path.name} has fixed_W_seed={fixed_seed}. "
            "You are not plotting the same W used in training."
        )
    if cfg_n_agg is not None and int(cfg_n_agg) != int(W_fixed.shape[1]):
        raise RuntimeError(
            f"Cluster mismatch: results.json n_agg={cfg_n_agg} but W_fixed has M={int(W_fixed.shape[1])}."
        )
    if int(W_fixed.shape[0]) != int(cfg_n_states):
        raise RuntimeError(
            f"State mismatch: results.json implies S={cfg_n_states} (b_max+s_max+1) but W_fixed has S={int(W_fixed.shape[0])}."
        )
    # Optional metadata checks from the payload (if present).
    if w_n_states is not None and int(w_n_states) != int(cfg_n_states):
        raise RuntimeError(f"W payload n_states={w_n_states} but results.json implies {cfg_n_states}.")
    if w_n_actions is not None and int(w_n_actions) != int(cfg_n_actions):
        raise RuntimeError(f"W payload n_actions={w_n_actions} but results.json implies {cfg_n_actions}.")
    if w_n_clusters is not None and cfg_n_agg is not None and int(w_n_clusters) != int(cfg_n_agg):
        raise RuntimeError(f"W payload n_clusters={w_n_clusters} but results.json n_agg={cfg_n_agg}.")

    row_sums = W_fixed.sum(dim=1)
    print(f"[W] loaded from {w_path}  seed={fixed_seed}  shape={tuple(W_fixed.shape)}  row_sum min/max=({row_sums.min():.6f},{row_sums.max():.6f})")

    lambda_ref, price_ref, ref_info = _resolve_reference_values(
        cfg,
        ref_split=args.reference_split,
        ref_index=int(args.reference_index),
        lambda_ref_override=args.lambda_ref,
        price_ref_override=args.price_ref,
    )
    ref_msg = (
        f"[Ref] source={ref_info['source']} split={ref_info['split']} "
        f"ref_index={ref_info['ref_index']} sample_index={ref_info['sample_index']} "
        f"lambda_ref={lambda_ref:.6f} price_ref={price_ref:.6f}"
    )
    if ref_info["p_related_ref"] is not None:
        ref_msg += f" p_related_ref={ref_info['p_related_ref']:.6f}"
    print(ref_msg)
    print(f"[Scan] raw lambda range=[{float(args.lam_min):.4f}, {float(args.lam_max):.4f}] with lambda_floor={float(args.lambda_floor):.4f}")

    lam_grid = np.linspace(float(args.lam_min), float(args.lam_max), int(args.n_grid))
    L_grid = np.zeros_like(lam_grid, dtype=float)
    t0 = time.time()
    for i, lam in enumerate(lam_grid):
        L_grid[i] = loss_landscape_forward(
            lam_val=float(lam),
            W_fixed=W_fixed,
            args_cfg=cfg,
            lambda_ref=lambda_ref,
            price_ref=price_ref,
            lambda_floor=float(args.lambda_floor),
        )
    print(f"[Landscape] computed {len(lam_grid)} points in {time.time()-t0:.1f}s")
    lam_min = float(lam_grid[np.argmin(L_grid)])
    L_min = float(L_grid.min())
    print(f"[Landscape] min at λ≈{lam_min:.4f}, L≈{L_min:.3f}")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        raise RuntimeError("matplotlib is required for plotting. Install with `pip install matplotlib`.") from e

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.3))
    ax.plot(lam_grid, L_grid, "k-", lw=1.8, alpha=0.85)
    ax.axvline(lambda_ref, color="red", ls="--", lw=1.2, label=f"λ_ref={lambda_ref:.2f}")
    ax.axvline(lam_min, color="#1f77b4", ls=":", lw=1.2, label=f"scan_min≈{lam_min:.2f}")
    if float(args.lam_min) < float(args.lambda_floor) < float(args.lam_max):
        ax.axvline(float(args.lambda_floor), color="gray", ls="-.", lw=1.0, label=f"λ_floor={args.lambda_floor:g}")
    ax.set_xlabel("λ")
    ax.set_ylabel("L(λ)")
    ax.set_title(
        f"Landscape with saved fixed W (seed={fixed_seed})\n"
        f"{results_dir.name} | p_ref={price_ref:.3f}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()

    pdf = out_dir / "fixed_W_landscape.pdf"
    plt.savefig(pdf, bbox_inches="tight")
    print(f"[Saved] {pdf}")


if __name__ == "__main__":
    main()
