"""Compute Cliff Markov geometry in one fresh process per grid scale.

The Cliff MDP stores its grid tables as module globals, so basic (48 states)
and medium (480 states) cannot be rebuilt safely in one notebook kernel.
This script reads saved aggregation checkpoints only; it never trains a model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
MARKOV_DIR = REPO / "notebooks" / "markov_aggregation"
SCRIPTS = MARKOV_DIR / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from mdp_lp.soft_state_aggregation import SoftAggDense
import markov_protocol_runner as base


ADDITIVE_ROOT = MARKOV_DIR / "results" / "additive"
NORMAL_ROOT = MARKOV_DIR / "results" / "ordinary_controls"


def load_state(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("state_dict", payload)


def standardized_distance(value: torch.Tensor) -> np.ndarray:
    array = value.detach().cpu().numpy().astype(float)
    array = (array - array.mean(0, keepdims=True)) / np.maximum(array.std(0, keepdims=True), 1e-8)
    return ((array[:, None, :] - array[None, :, :]) ** 2).mean(-1)


def entropy(W: np.ndarray) -> float:
    return float(-(W * np.log(np.maximum(W, 1e-12))).sum(1).mean() / np.log(W.shape[1]))


def cliff_probe(seed: int, scale: str, output: Path):
    if scale == "medium":
        from cliff_large_state_impl import configure_cliff_grid
        configure_cliff_grid(n_rows=4, n_cols=120)
    args = base._cliff_args("sa", base.CLIFF_RAW_SEEDS[seed], output, scale, n_agg=12)
    _, test_loader, _ = base._make_cliff_data(args, base._cliff_setup(scale))
    sample = next(iter(test_loader))
    P, r = sample[3][:1], sample[4][:1]
    gamma = torch.zeros(1, P.shape[1], dtype=P.dtype)
    gamma[:, 0] = 1.0
    return args, P, r, gamma


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=("basic", "medium"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args_cli = parser.parse_args()

    records = [
        row for row in json.loads((ADDITIVE_ROOT / "summary.json").read_text(encoding="utf-8"))["records"]
        if row.get("status") == "ok" and not row.get("screen")
        and row.get("task") == "cliff" and row.get("scale") == args_cli.scale
        and row.get("method") in {"markov_detach_m12", "markov_e2e_m12", "markov_scaled_m12"}
    ]
    rows = []
    for row in records:
        seed = int(row["seed_index"])
        normal_dir = NORMAL_ROOT / "cliff" / args_cli.scale / "sa_m12" / f"seed_{seed}"
        run_args, P, r, gamma = cliff_probe(seed, args_cli.scale, normal_dir)
        tau = float(row.get("protocol_config", {}).get("args", {}).get("sa_tau", 0.5))

        normal = SoftAggDense(P.shape[1], P.shape[2], 12, r_embed=run_args.sa_r_embed)
        normal.load_state_dict(load_state(normal_dir / "aggregation_state.pt"))
        with torch.no_grad():
            _, _, _, normal_info = normal(P, r, gamma, tau=tau)
        base_feature = torch.cat([
            torch.einsum("bsaj,jr->bsar", P, normal.dest_embed.weight).reshape(1, P.shape[1], -1), r
        ], dim=-1)[0]
        W_normal = normal_info["W"][0].detach().cpu().numpy()

        spec = base.MarkovSpec(**row["spec"])
        route = row["method"]
        path = route.rsplit("_m", 1)[0]
        markov = base.ProtocolMarkovSoftAgg(
            n_states=P.shape[1], n_actions=P.shape[2], n_clusters=12,
            r_embed=run_args.sa_r_embed, spec=spec, path=path,
        )
        state_path = ADDITIVE_ROOT / "cliff" / args_cli.scale / route / f"seed_{seed}" / "aggregation_state.pt"
        markov.load_state_dict(load_state(state_path))
        with torch.no_grad():
            _, _, _, markov_info = markov(P, r, gamma, tau=tau)
            h = markov.signature_encoder(markov._signature(P, r))[0]
        W_markov = markov_info["W"][0].detach().cpu().numpy()
        upper = np.triu_indices(P.shape[1], k=1)
        rep_similarity = float(spearmanr(
            standardized_distance(base_feature)[upper], standardized_distance(h)[upper]
        ).statistic)
        delta_c = float(np.abs((W_markov @ W_markov.T) - (W_normal @ W_normal.T))[upper].mean())
        rows.append({
            "task": "cliff", "scale": args_cli.scale, "M": 12, "seed": seed,
            "implementation": "additive / " + route.removeprefix("markov_").removesuffix("_m12"),
            "ordinary-Markov representation similarity": rep_similarity,
            "Markov_entropy": entropy(W_markov), "Delta_C_vs_ordinary": delta_c,
        })

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"scale": args_cli.scale, "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
