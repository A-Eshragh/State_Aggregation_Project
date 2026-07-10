"""
Utilities to interpret saved cliff runs (run_config.json, loaders, skew helpers).
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split

from mdp_cw.prepare_cliff_data import CliffDataConfig, prepare_cliff_data
from mdp_cw.weather_dynamics import (
    REGIME_NAMES_V1,
    REGIME_NAMES_V2,
    REGIME_NAMES_V3,
    REGIME_NAMES_V4,
    sample_regimes,
    set_regime_set,
)


def regime_names_for_set(regime_set: str) -> list[str]:
    rs = (regime_set or "v1").lower()
    if rs == "v1":
        return list(REGIME_NAMES_V1)
    if rs == "v2":
        return list(REGIME_NAMES_V2)
    if rs == "v3":
        return list(REGIME_NAMES_V3)
    if rs == "v4":
        return list(REGIME_NAMES_V4)
    return [f"regime_{i}" for i in range(16)]


def is_dose_style_skew(regime_weights: list[float]) -> bool:
    """True for v3 dose weights ``k,1,1,1`` with k > 1 (last three equal, first larger)."""
    if len(regime_weights) != 4:
        return False
    w0, w1, w2, w3 = regime_weights
    if abs(w1 - w2) > 1e-6 or abs(w2 - w3) > 1e-6:
        return False
    return w0 > w1 + 1e-9


def scale_loss_weight_for_skew(
    loss_weight: float,
    regime_weights: Optional[list[float]],
    *,
    p_ice_ref: float = 0.3,
) -> float:
    """Up-weight CE when ICE regimes are rare (DFL-only scheduling).

    For weights ``k,1,1,1``, p_ice = 3/(k+3). With p_ice_ref = 0.3 (= 3/10, k=7),
    effective weight scales like (k+3)/10 relative to the k=7 baseline.
    """
    if regime_weights is None or not is_dose_style_skew(regime_weights):
        return float(loss_weight)
    s = float(sum(regime_weights))
    p_ice = float(sum(regime_weights[1:])) / max(s, 1e-12)
    if p_ice < 1e-9:
        return float(loss_weight)
    return float(loss_weight) * (p_ice_ref / p_ice)


def read_run_config(run_dir: Path) -> dict[str, str]:
    path = run_dir / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text())


def cliff_data_config_from_run_config(rc: dict[str, str]) -> CliffDataConfig:
    ambiguous_pairs = None
    ap = (rc.get("ambiguous_pairs") or "").strip()
    if ap:
        ambiguous_pairs = []
        for tok in ap.split(","):
            i, j = tok.strip().split(":")
            ambiguous_pairs.append((int(i), int(j)))

    regime_weights = None
    rw = (rc.get("regime_weights") or "").strip()
    if rw:
        regime_weights = [float(x) for x in rw.split(",")]
    test_regime_weights = None
    trw = (rc.get("test_regime_weights") or "").strip()
    if trw:
        test_regime_weights = [float(x) for x in trw.split(",")]

    return CliffDataConfig(
        n_samples=int(float(rc["n_samples"])),
        feat_dim=int(float(rc["data_feat_dim"])),
        noise_std=float(rc["noise_std"]),
        beta=float(rc["beta"]),
        train_frac=float(rc["train_frac"]),
        batch_size=int(float(rc["batch_size"])),
        seed=int(float(rc["seed"])),
        regime_set=str(rc.get("regime_set", "v1")),
        center_scale=float(rc.get("center_scale", 2.0)),
        ambiguous_pairs=ambiguous_pairs,
        cliff_penalty=float(rc["cliff_penalty"]),
        regime_weights=regime_weights,
        test_regime_weights=test_regime_weights,
    )


def test_regime_index_order(run_dir: Path) -> np.ndarray:
    """Latent regime id per test example (same order as ``test_loader``), cheaply.

    Avoids rebuilding full MDP tensors / oracle LPs — only reproduces ``z`` and
    the ``random_split`` from ``prepare_cliff_data``.
    """
    rc = read_run_config(run_dir)
    cfg = cliff_data_config_from_run_config(rc)
    set_regime_set(cfg.regime_set)
    torch.manual_seed(cfg.seed)
    z = sample_regimes(cfg.n_samples, seed=cfg.seed, weights=cfg.regime_weights)
    ds = TensorDataset(z.long())
    gen = torch.Generator().manual_seed(cfg.seed + 7)
    n_train = max(1, int(cfg.train_frac * cfg.n_samples))
    n_test = cfg.n_samples - n_train
    if n_test == 0:
        return z.detach().cpu().numpy().astype(np.int64)
    _tr, te = random_split(ds, [n_train, n_test], generator=gen)
    ind = np.asarray(te.indices, dtype=np.int64)
    return z.detach().cpu().numpy()[ind].astype(np.int64)


def prepare_loaders_from_run_dir(run_dir: Path):
    """Rebuild (train_loader, test_loader, meta) exactly as in the original run."""
    rc = read_run_config(run_dir)
    cfg = cliff_data_config_from_run_config(rc)
    return prepare_cliff_data(cfg)


def namespace_from_run_config(rc: dict[str, str]) -> Namespace:
    """Minimal Namespace for ``_solve_with_algo`` / model construction."""
    return Namespace(
        algo=str(rc["algo"]),
        beta=float(rc["beta"]),
        qp_reg=float(rc.get("qp_reg", 0.1)),
        feat_dim=int(float(rc["nn_feat_dim"])),
        hidden_dim=int(float(rc["hidden_dim"])),
        pretrain_epochs=int(float(rc.get("pretrain_epochs", 0))),
        lr=float(rc["lr"]),
        n_epochs=int(float(rc.get("n_epochs", 0))),
        grad_clip=float(rc.get("grad_clip", 10.0)),
        n_agg=int(float(rc["n_agg"])) if rc.get("n_agg") else 12,
        sa_tau=float(rc["sa_tau"]) if rc.get("sa_tau") else 0.5,
        sa_r_embed=int(float(rc["sa_r_embed"])) if rc.get("sa_r_embed") else 8,
    )


__all__ = [
    "cliff_data_config_from_run_config",
    "is_dose_style_skew",
    "namespace_from_run_config",
    "prepare_loaders_from_run_dir",
    "read_run_config",
    "regime_names_for_set",
    "scale_loss_weight_for_skew",
    "test_regime_index_order",
]
