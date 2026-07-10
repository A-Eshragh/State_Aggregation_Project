"""
CLI entry point for cliff-walking experiments.

Example:

    python -m mdp_cw.run_cliff_experiments \
        --algo dfl \
        --n_samples 200 \
        --n_epochs 50 \
        --pretrain_epochs 60 \
        --save_dir mdp_cw/results/dfl_run0

Algorithms:
    twostage   — supervised CE on slip distributions (no LP in training).
    dfl        — differentiable LP via basis-sensitivity.
    dfl_qpth   — qpth-based QP-regularized surrogate.
    dfl_ssa    — DFL on a learned soft state aggregation (S=48 -> M clusters).
    dfl_sketch — sketched LP: average of K row-reduced LPs (see --n_sketches, --sketch_ratio).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mdp_cw.cliff_run_utils import scale_loss_weight_for_skew
from mdp_cw.prepare_cliff_data import CliffDataConfig, prepare_cliff_data
from mdp_cw.train_cliff import train_eval_cliff


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, default="dfl",
                   choices=["twostage", "dfl", "dfl_qpth", "dfl_ssa", "dfl_sketch"])

    # Data
    p.add_argument("--n_samples", type=int, default=200)
    p.add_argument("--feat_dim", type=int, default=8,
                   help="Dimension of raw weather features w.")
    p.add_argument("--noise_std", type=float, default=0.6,
                   help="Std of Gaussian noise added to per-regime feature centers.")
    p.add_argument("--train_frac", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--regime_set", type=str, default="v1", choices=["v1", "v2", "v3", "v4", "v5", "v6"],
                   help="v1 = 4 regimes (original, state-independent); "
                        "v2 = 8 policy-diverse regimes (state-independent); "
                        "v3 = 4 regimes, 3 volatile cells (state-dependent); "
                        "v4 = 4 regimes, 6 volatile cells / richer ICE patterns (harder); "
                        "v5 = 4 regimes, shared row-2 hazard strip (decision-sharp); "
                        "v6 = 4 regimes, shared late gate trap.")
    p.add_argument("--center_scale", type=float, default=2.0,
                   help="Radius of regime cluster centers in feature space (smaller = more overlap).")
    p.add_argument("--ambiguous_pairs", type=str, default="",
                   help="Comma-separated i:j pairs of regime ids whose centers should be "
                        "pinned near-identical, e.g. '3:4' makes regimes 3 and 4 visually ambiguous.")
    p.add_argument("--ambiguous_gap", type=float, default=0.15,
                   help="Cluster-center offset for ambiguous pairs (default 0.15). "
                        "Set to 0.0 to make paired centers statistically identical.")
    p.add_argument("--cliff_penalty", type=float, default=-100.0,
                   help="Cliff reward (gym default = -100). Smaller magnitude gives a soft-cliff "
                        "variant (not gym-equivalent) with smoother reward landscape, useful for "
                        "DFL stability.")
    p.add_argument("--regime_weights", type=str, default="",
                   help="Comma-separated non-negative weights (one per regime). Normalised internally. "
                        "Empty = uniform. E.g. '0.7,0.1,0.1,0.1' skews the prior on v3.")
    p.add_argument("--test_regime_weights", type=str, default="",
                   help="Optional test-only regime weights. If set, train samples use --regime_weights "
                        "and test samples use this prior instead of a random split from one pooled prior.")

    # Model
    p.add_argument("--hidden_dim", type=int, default=64,
                   help="Hidden layer width of the FeatureNet encoder.")
    p.add_argument("--nn_feat_dim", dest="nn_feat_dim", type=int, default=32,
                   help="Dimension of the frozen feature vector h used by the head.")

    # Optim
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--pretrain_lr", type=float, default=1e-2)
    p.add_argument("--pretrain_epochs", type=int, default=60)
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=10.0)

    # DFL / QP
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--qp_reg", type=float, default=0.1)
    p.add_argument(
        "--qp_max_iter",
        type=int,
        default=20,
        help="qpth PDIPM max iterations (dfl_qpth only; higher can improve policy quality at more wall time).",
    )
    p.add_argument("--loss_weight", type=float, default=1.0,
                   help="Weight on the slip_CE prediction loss added to DFL loss.")
    p.add_argument(
        "--scale_loss_weight_with_skew",
        action="store_true",
        help="DFL / dfl_qpth / dfl_ssa only: multiply loss_weight by p_ice_ref/p_ice when "
             "regime_weights are dose-style k,1,1,1 on v3 (helps high-skew runs).",
    )
    p.add_argument(
        "--loss_weight_skew_p_ice_ref",
        type=float,
        default=0.3,
        help="Reference mass on {ice_west, ice_mid, ice_east} under k=7 (default 3/10).",
    )
    p.add_argument("--aug_lag_rho", type=float, default=0.1)
    p.add_argument(
        "--dfl_perturb_std",
        type=float,
        default=0.0,
        help="PFYL noise std added to r_pred before LP solve in dfl mode (0 = disabled).",
    )
    p.add_argument(
        "--qpth_warmup_epochs",
        type=int,
        default=0,
        help="For dfl algo: use QPTH solver for this many epochs before switching to "
             "fixed-basis LP. QPTH gradient moves model into correct LP basis; DFL then refines.",
    )

    # Soft state aggregation (only used by --algo dfl_ssa)
    p.add_argument("--n_agg", type=int, default=12,
                   help="Number of soft clusters M in dfl_ssa (S=48 -> M).")
    p.add_argument("--sa_tau", type=float, default=0.5,
                   help="Softmax temperature for soft cluster memberships.")
    p.add_argument("--sa_r_embed", type=int, default=8,
                   help="Destination-state embedding dim for SoftAggDense.")

    # Sketched LP (used by --algo dfl_sketch and by --algo dfl_ssa as the
    # smoother on top of the aggregated LP; mirrors mdp_lp's dfl_sketch+n_agg).
    p.add_argument("--n_sketches", type=int, default=4,
                   help="Number of random sketches K to average. dfl_sketch: try 4 or 8. dfl_ssa: try 4-8.")
    p.add_argument("--sketch_ratio", type=float, default=0.1,
                   help="Fraction of LP rows to keep per sketch. dfl_sketch: try 0.1-0.2. dfl_ssa: try 0.5-0.7 (M is small).")
    p.add_argument("--sketch_mode", type=str, default="gaussian",
                   choices=["gaussian", "rademacher", "subsample"],
                   help="Sketch matrix distribution.")

    # IO
    p.add_argument("--save_dir", type=str, default=None)

    return p.parse_args()


def _map_feat_dim(args):
    """feat_dim in the model wants to be the hidden feature size (nn_feat_dim),
    not the raw weather dim. Make the mapping explicit so train_cliff can read it."""
    args.feat_dim_model = args.nn_feat_dim
    return args


def main():
    args = parse_args()
    # Keep raw feature dim for the data generator under a different name.
    args.data_feat_dim = args.feat_dim

    torch.manual_seed(args.seed)

    ambiguous_pairs = None
    if args.ambiguous_pairs.strip():
        ambiguous_pairs = []
        for tok in args.ambiguous_pairs.split(","):
            i, j = tok.split(":")
            ambiguous_pairs.append((int(i), int(j)))

    regime_weights = None
    if args.regime_weights.strip():
        regime_weights = [float(x) for x in args.regime_weights.split(",")]
    test_regime_weights = None
    if args.test_regime_weights.strip():
        test_regime_weights = [float(x) for x in args.test_regime_weights.split(",")]

    cfg = CliffDataConfig(
        n_samples=args.n_samples,
        feat_dim=args.data_feat_dim,
        noise_std=args.noise_std,
        beta=args.beta,
        train_frac=args.train_frac,
        batch_size=args.batch_size,
        seed=args.seed,
        regime_set=args.regime_set,
        center_scale=args.center_scale,
        ambiguous_pairs=ambiguous_pairs,
        ambiguous_gap=args.ambiguous_gap,
        cliff_penalty=args.cliff_penalty,
        regime_weights=regime_weights,
        test_regime_weights=test_regime_weights,
    )
    train_loader, test_loader, meta = prepare_cliff_data(cfg)

    # Training expects args.feat_dim to be the frozen feature dim.
    args.feat_dim = args.nn_feat_dim

    if (
        args.scale_loss_weight_with_skew
        and args.algo != "twostage"
        and regime_weights is not None
    ):
        args.loss_weight = scale_loss_weight_for_skew(
            float(args.loss_weight),
            regime_weights,
            p_ice_ref=float(args.loss_weight_skew_p_ice_ref),
        )

    if args.save_dir is not None:
        args.save_dir = Path(args.save_dir)
        args.save_dir.mkdir(parents=True, exist_ok=True)
        (args.save_dir / "run_config.json").write_text(
            json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2)
        )

    result = train_eval_cliff(args, train_loader, test_loader, meta)
    return result


if __name__ == "__main__":
    main()
