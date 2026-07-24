"""Self-training runner for the Markov-aggregation notebook protocol.

This module deliberately starts every reported run from generated data and a
fresh model.  It never reads prior experiment summaries as an input.  The
notebook calls this module in small isolated worker processes so the 48-state
and widened Cliff constructors cannot leak global MDP constants into one
another.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import re
import subprocess
import sys
import time
import traceback
from argparse import Namespace
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch import nn
from torch.utils.data import DataLoader, random_split

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
AGG_DIR = Path(__file__).resolve().parent
if str(AGG_DIR) not in sys.path:
    sys.path.insert(0, str(AGG_DIR))

from markov_signature_aggregation import aggregate_with_membership, choose_even_actions, compute_markov_signature
from mdp_lp.soft_state_aggregation import SoftAggDense
from resource_profile import ResourceMonitor


# These are raw data/model seeds, not a post-hoc selected list.
CLIFF_RAW_SEEDS = [360, 606, 689, 665, 790, 310, 582, 962, 246, 297]
INVENTORY_RAW_SEEDS = list(range(1, 11))
FINAL_PATHS = ("markov_e2e", "markov_detach", "markov_scaled")
BASELINES = ("ts", "lp", "qp", "sketch", "sa_m12", "sa_m24")
RUNNER_SCHEMA_VERSION = "markov_protocol_v5"

# This runner trains only newly proposed Markov variants.  Ordinary-SA and
# all other baselines are fixed, read-only records in results_baseline.
ACTIVE_CELLS = (
    ("inventory", "basic"),
    ("cliff", "basic"),
    ("cliff", "medium"),
)

# The fixed normal-SA comparison uses these aggregation sizes in each active
# cell.  Inventory medium is deliberately deferred until its ordinary SA
# passes a decision-usefulness gate against TS.
MARKOV_SIZES = {
    ("inventory", "basic"): (10, 20, 30),
    ("cliff", "basic"): (12,),
    ("cliff", "medium"): (12,),
}


def _method_n_agg(method: str, default: int = 12) -> int:
    match = re.search(r"_m(\d+)$", method)
    return int(match.group(1)) if match else int(default)


@dataclass(frozen=True)
class MarkovSpec:
    depth: int = 1
    n_word_actions: int = 3
    n_terminal_actions: int = 3
    use_raw_state_index: bool = False
    hidden: int = 24
    embed: int = 12
    alpha: float = 0.15
    gradient_scale: float = 0.10
    n_agg: int = 12
    tau: float = 0.5


SCREEN_SPECS = (
    MarkovSpec(depth=1, n_word_actions=3, n_terminal_actions=3, hidden=24, embed=12, alpha=0.15, gradient_scale=0.05),
    MarkovSpec(depth=1, n_word_actions=3, n_terminal_actions=3, hidden=24, embed=12, alpha=0.15, gradient_scale=0.15),
    MarkovSpec(depth=2, n_word_actions=2, n_terminal_actions=3, hidden=24, embed=12, alpha=0.10, gradient_scale=0.05),
    MarkovSpec(depth=2, n_word_actions=2, n_terminal_actions=3, hidden=24, embed=12, alpha=0.10, gradient_scale=0.15),
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _protocol_config(
    *,
    task: str,
    scale: str,
    method: str,
    raw_seed: int,
    screen: bool,
    spec: "MarkovSpec | None",
    n_agg: int,
    args: Namespace,
) -> dict[str, Any]:
    """Persist the settings that make a raw worker safe to reuse.

    This deliberately records the protocol knobs rather than a broad dump of
    ``Namespace`` so a cache mismatch is intelligible in the notebook.
    """
    arg_names = (
        "algo", "n_samples", "batch_size", "train_batch_size", "eval_batch_size", "n_epochs", "pretrain_epochs", "lr", "pretrain_lr",
        "beta", "loss_weight", "aug_lag_rho", "qp_reg", "qp_backend", "n_sketches",
        "sketch_ratio", "sketch_mode", "sa_tau", "agg_lr_ratio", "grad_clip", "qpth_warmup_epochs",
        "structured_qp_tol", "structured_qp_max_iter", "s_max", "b_max", "q_max", "reward_eval_batch_size",
        "use_aggregated_eval_policy", "use_cached_eval_oracle", "skip_initial_reward_eval",
        "use_aggregated_train_reward_eval",
    )
    return {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "task": task,
        "scale": scale,
        "method": method,
        "raw_seed": int(raw_seed),
        "screen": bool(screen),
        "evaluation_split": "validation" if screen else "held_out_test",
        "validation_split_seed": int(raw_seed + 71) if screen else None,
        "n_agg": int(n_agg),
        "markov_spec": asdict(spec) if spec else None,
        "cuda_available": bool(torch.cuda.is_available()),
        "cpu_threads_per_worker": int(os.environ.get("MARKOV_PROTOCOL_CPU_THREADS", "8")),
        "retry_policy": "fresh_worker; reuse only matching protocol fingerprint; force recreates worker",
        "args": {name: _jsonable(getattr(args, name)) for name in arg_names if hasattr(args, name)},
        **({"cliff_evaluation_batch_size": int(os.environ.get("MARKOV_PROTOCOL_CLIFF_EVAL_BATCH_SIZE", "8"))} if task == "cliff" else {}),
    }


def _protocol_fingerprint(config: dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output * ctx.scale, None


def scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    return _ScaleGradient.apply(x, float(scale))


class ProtocolMarkovSoftAgg(nn.Module):
    """One common forward representation with three backward treatments.

    ``e2e`` passes the signature gradient to predicted dynamics; ``detach``
    stops only that path before its learned head; ``scaled`` keeps the exact
    e2e forward value but bounds the signature gradient by ``gradient_scale``.
    The ordinary learned SoftAgg base logits remain present in all three.
    """

    def __init__(
        self,
        *,
        n_states: int,
        n_actions: int,
        n_clusters: int,
        r_embed: int = 8,
        spec: MarkovSpec,
        path: str,
        registry: list["ProtocolMarkovSoftAgg"] | None = None,
        **_: Any,
    ):
        super().__init__()
        if path not in FINAL_PATHS:
            raise ValueError(f"Unknown Markov path: {path}")
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.n_clusters = int(n_clusters)
        self.r_embed = int(r_embed)
        self.spec = spec
        self.path = path
        self.dest_embed = nn.Embedding(self.n_states, self.r_embed)
        self.to_logits = nn.Linear(self.n_actions * self.r_embed, self.n_clusters)
        self.r_bias = nn.Linear(self.n_actions, self.n_clusters, bias=False)
        nn.init.normal_(self.r_bias.weight, std=1e-3)
        self.word_actions = choose_even_actions(self.n_actions, spec.n_word_actions)
        self.terminal_actions = choose_even_actions(self.n_actions, spec.n_terminal_actions)
        n_words = sum(len(self.word_actions) ** d for d in range(spec.depth + 1))
        self.signature_dim = n_words * len(self.terminal_actions) + int(spec.use_raw_state_index)
        self.signature_encoder = nn.Sequential(
            nn.Linear(self.signature_dim, spec.hidden), nn.ReLU(), nn.Linear(spec.hidden, spec.embed), nn.ReLU()
        )
        self.markov_to_logits = nn.Linear(spec.embed, self.n_clusters)
        nn.init.normal_(self.markov_to_logits.weight, std=1e-2)
        nn.init.zeros_(self.markov_to_logits.bias)
        self.logit_gradient_norms: dict[str, list[float]] = {
            "base_logits": [], "markov_logits": [], "signature": [], "transition": [], "reward": [],
        }
        self._initial_state = {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}
        self._initial_probe: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._initial_w: torch.Tensor | None = None
        if registry is not None:
            registry.append(self)

    @staticmethod
    def disaggregate_y(y_agg: torch.Tensor, W: torch.Tensor, n_actions: int) -> torch.Tensor:
        return SoftAggDense.disaggregate_y(y_agg, W, n_actions)

    def _base_logits(self, P: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        phi = torch.einsum("bsaj,jr->bsar", P, self.dest_embed.weight)
        return self.to_logits(phi.reshape(P.shape[0], P.shape[1], -1)) + self.r_bias(r)

    def _signature(self, P: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        signature = compute_markov_signature(
            P, r, max_depth=self.spec.depth,
            word_actions=self.word_actions, terminal_actions=self.terminal_actions,
        )
        signature = (signature - signature.mean(dim=1, keepdim=True)) / signature.std(dim=1, keepdim=True).clamp_min(1e-6)
        if self.spec.use_raw_state_index:
            idx = torch.linspace(0.0, 1.0, P.shape[1], device=P.device, dtype=P.dtype)
            signature = torch.cat([signature, idx.view(1, -1, 1).expand(P.shape[0], -1, 1)], dim=-1)
        if torch.is_grad_enabled() and signature.requires_grad:
            signature.register_hook(lambda grad: self.logit_gradient_norms["signature"].append(float(grad.norm().detach().cpu())))
        if self.path == "markov_detach":
            signature = signature.detach()
        elif self.path == "markov_scaled":
            signature = scale_gradient(signature, self.spec.gradient_scale)
        return signature

    def compute_logits(self, P: torch.Tensor, r: torch.Tensor):
        if torch.is_grad_enabled() and P.requires_grad:
            P.register_hook(lambda grad: self.logit_gradient_norms["transition"].append(float(grad.norm().detach().cpu())))
        if torch.is_grad_enabled() and r.requires_grad:
            r.register_hook(lambda grad: self.logit_gradient_norms["reward"].append(float(grad.norm().detach().cpu())))
        base = self._base_logits(P, r)
        markov = self.markov_to_logits(self.signature_encoder(self._signature(P, r)))
        markov = (markov - markov.mean(dim=(1, 2), keepdim=True)) / markov.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        if torch.is_grad_enabled() and base.requires_grad:
            base.register_hook(lambda g: self.logit_gradient_norms["base_logits"].append(float(g.norm().detach().cpu())))
        if torch.is_grad_enabled() and markov.requires_grad:
            markov.register_hook(lambda g: self.logit_gradient_norms["markov_logits"].append(float(g.norm().detach().cpu())))
        return base + self.spec.alpha * markov, base, markov

    def forward(self, P: torch.Tensor, r: torch.Tensor, gamma: torch.Tensor, tau: float = 0.5):
        start = time.perf_counter()
        logits, base, markov = self.compute_logits(P, r)
        W = F.softmax(logits / max(float(tau), 1e-8), dim=-1)
        if self._initial_probe is None:
            self._initial_probe = (P.detach().cpu().clone(), r.detach().cpu().clone(), gamma.detach().cpu().clone())
            self._initial_w = W[0].detach().cpu().clone()
        P_agg, r_agg, gamma_agg, mass = aggregate_with_membership(P, r, gamma, W)
        return P_agg, r_agg, gamma_agg, {
            "W": W, "mass": mass, "logits": logits, "base_logits": base,
            "markov_logits": markov, "runtime_sec": time.perf_counter() - start,
        }

    @staticmethod
    def _distribution(values: list[float]) -> dict[str, Any]:
        data = np.asarray(values, dtype=float)
        if not data.size:
            return {"n": 0, "mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "values": []}
        return {
            "n": int(data.size), "mean": float(data.mean()), "std": float(data.std(ddof=0)),
            "p50": float(np.quantile(data, 0.50)), "p90": float(np.quantile(data, 0.90)),
            "p99": float(np.quantile(data, 0.99)),
            "values": data.tolist(),
        }

    def gradient_summary(self) -> dict[str, Any]:
        stats = {name: self._distribution(values) for name, values in self.logit_gradient_norms.items()}
        base = float(stats["base_logits"]["mean"])
        markov = float(stats["markov_logits"]["mean"])
        return {
            "by_tensor": stats,
            "markov_to_base_logit_grad_ratio": markov / max(base, 1e-12),
            "signature_to_transition_grad_ratio": float(stats["signature"]["mean"]) / max(float(stats["transition"]["mean"]), 1e-12),
        }

    def membership_movement(self, tau: float) -> dict[str, float]:
        if self._initial_probe is None or self._initial_w is None:
            return {"mean_l1": float("nan"), "mean_l2": float("nan"), "hard_label_change_rate": float("nan")}
        device = next(self.parameters()).device
        P, r, gamma = (value.to(device) for value in self._initial_probe)
        with torch.no_grad():
            logits, _, _ = self.compute_logits(P, r)
            final_w = F.softmax(logits / max(float(tau), 1e-8), dim=-1)[0].detach().cpu()
        delta = final_w - self._initial_w
        return {
            "mean_l1": float(delta.abs().sum(dim=-1).mean()),
            "mean_l2": float(delta.pow(2).sum(dim=-1).sqrt().mean()),
            "hard_label_change_rate": float((final_w.argmax(dim=-1) != self._initial_w.argmax(dim=-1)).float().mean()),
        }

    def parameter_update_summary(self) -> dict[str, float]:
        groups = {
            "base": ("dest_embed", "to_logits", "r_bias"),
            "markov": ("signature_encoder", "markov_to_logits"),
        }
        result = {}
        state = self.state_dict()
        for group, prefixes in groups.items():
            delta_sq = 0.0
            initial_sq = 0.0
            for name, final in state.items():
                if not name.startswith(prefixes):
                    continue
                initial = self._initial_state[name]
                delta_sq += float((final.detach().cpu() - initial).pow(2).sum())
                initial_sq += float(initial.pow(2).sum())
            result[f"{group}_parameter_update_l2"] = float(math.sqrt(delta_sq))
            result[f"{group}_parameter_relative_update"] = float(math.sqrt(delta_sq) / max(math.sqrt(initial_sq), 1e-12))
        return result

    def trainable_definition(self) -> dict[str, list[str]]:
        return {
            "base_membership": ["dest_embed", "to_logits", "r_bias"],
            "markov_membership": ["signature_encoder", "markov_to_logits"],
            "signature_to_predictor_gradient": ["present" if self.path != "markov_detach" else "detached"],
        }


def _make_protocol_agg(spec: MarkovSpec, path: str, registry: list[ProtocolMarkovSoftAgg]):
    class ConfiguredProtocolMarkovAgg(ProtocolMarkovSoftAgg):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, spec=spec, path=path, registry=registry, **kwargs)
    return ConfiguredProtocolMarkovAgg


def _make_capture_normal_agg(registry: list[SoftAggDense]):
    """Capture a normal SoftAgg instance without changing its forward rule."""
    class CapturedNormalSoftAgg(SoftAggDense):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self._initial_state = {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}
            self._initial_probe: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
            self._initial_w: torch.Tensor | None = None
            registry.append(self)

        def forward(self, P: torch.Tensor, r: torch.Tensor, gamma: torch.Tensor, tau: float = 0.5):
            outputs = super().forward(P, r, gamma, tau=tau)
            info = outputs[3]
            if self._initial_probe is None:
                self._initial_probe = (P.detach().cpu().clone(), r.detach().cpu().clone(), gamma.detach().cpu().clone())
                self._initial_w = info["W"][0].detach().cpu().clone()
            return outputs

        def membership_movement(self, tau: float) -> dict[str, float]:
            if self._initial_probe is None or self._initial_w is None:
                return {"mean_l1": float("nan"), "mean_l2": float("nan"), "hard_label_change_rate": float("nan")}
            device = next(self.parameters()).device
            P, r, gamma = (value.to(device) for value in self._initial_probe)
            with torch.no_grad():
                _, _, _, info = super().forward(P, r, gamma, tau=tau)
                final_w = info["W"][0].detach().cpu()
            delta = final_w - self._initial_w
            return {
                "mean_l1": float(delta.abs().sum(dim=-1).mean()),
                "mean_l2": float(delta.pow(2).sum(dim=-1).sqrt().mean()),
                "hard_label_change_rate": float((final_w.argmax(dim=-1) != self._initial_w.argmax(dim=-1)).float().mean()),
            }

        def parameter_update_summary(self) -> dict[str, float]:
            delta_sq = 0.0
            initial_sq = 0.0
            for name, final in self.state_dict().items():
                initial = self._initial_state[name]
                delta_sq += float((final.detach().cpu() - initial).pow(2).sum())
                initial_sq += float(initial.pow(2).sum())
            return {
                "base_parameter_update_l2": float(math.sqrt(delta_sq)),
                "base_parameter_relative_update": float(math.sqrt(delta_sq) / max(math.sqrt(initial_sq), 1e-12)),
            }
    return CapturedNormalSoftAgg


@contextmanager
def _patch_aggregator(task: str, spec: MarkovSpec | None, path: str | None, registry: list[ProtocolMarkovSoftAgg]):
    if spec is None:
        yield
        return
    if task == "inventory":
        import mdp_lp.train_elasticity as train_mod
    else:
        import mdp_cw.train_cliff as train_mod
    old = train_mod.SoftAggDense
    train_mod.SoftAggDense = _make_protocol_agg(spec, path or "markov_e2e", registry)
    try:
        yield
    finally:
        train_mod.SoftAggDense = old


@contextmanager
def _capture_normal_aggregator(task: str, registry: list[SoftAggDense]):
    if task == "inventory":
        import mdp_lp.train_elasticity as train_mod
    else:
        import mdp_cw.train_cliff as train_mod
    old = train_mod.SoftAggDense
    train_mod.SoftAggDense = _make_capture_normal_agg(registry)
    try:
        yield
    finally:
        train_mod.SoftAggDense = old


def _cliff_setup(scale: str):
    if scale == "medium":
        from cliff_large_state_impl import configure_cliff_grid
        configure_cliff_grid(n_rows=4, n_cols=120)
    from mdp_cw.prepare_cliff_data import CliffDataConfig, prepare_cliff_data
    from mdp_cw import train_cliff
    return CliffDataConfig, prepare_cliff_data, train_cliff


def _cliff_state_count(scale: str) -> int:
    """Known constructor dimensions; never rebuild a dense test MDP for metadata."""
    return 48 if scale == "basic" else 4 * 120


def _cliff_args(method: str, seed: int, output: Path, scale: str, *, n_agg: int = 12) -> Namespace:
    # Medium keeps the same model and epoch schedule; only the MDP constructor changes.
    n_samples = 200 if scale == "basic" else 80
    # Basic Cliff follows the fixed reproduction configuration for the
    # M=12 control.  Medium is a separately labelled scale extension, where
    # a smaller reward-evaluation batch is required for numerical stability.
    fixed_basic = scale == "basic"
    return Namespace(
        algo={"ts": "twostage", "lp": "dfl", "qp": "dfl_qpth", "sketch": "dfl_sketch"}.get(method, "dfl_ssa"),
        n_samples=n_samples, data_feat_dim=8, feat_dim=32, noise_std=2.5, train_frac=0.5, batch_size=32,
        reward_eval_batch_size=(32 if fixed_basic else int(os.environ.get("MARKOV_PROTOCOL_CLIFF_REWARD_EVAL_BATCH_SIZE", "1"))),
        seed=seed, regime_set="v4", center_scale=0.88, ambiguous_pairs=[(0, 2)], ambiguous_gap=0.0,
        cliff_penalty=-50.0, regime_weights=[15.0, 1.0, 1.0, 1.0], test_regime_weights=[0.0, 0.0, 1.0, 0.0],
        hidden_dim=64, nn_feat_dim=32, lr=0.003, pretrain_lr=0.01, pretrain_epochs=60, n_epochs=50,
        grad_clip=10.0, beta=0.95, qp_reg=0.1, qp_max_iter=20, loss_weight=500.0, aug_lag_rho=0.1,
        dfl_perturb_std=0.0, qpth_warmup_epochs=0, n_agg=n_agg, sa_tau=0.5, sa_r_embed=8,
        n_sketches=4, sketch_ratio=0.1, ssa_objective="cw" if method.startswith("sa") or method.startswith("markov") else "none",
        cw_agg_mode="dest_r_bias", agg_lr_ratio=1.0, lp_basis_ridge=0.0, hinge_weight_alpha=0.0,
        hard_mine_every=0, margin_lambda=0.0, sketch_use_qpth=False, stochastic_membership_samples=1,
        stochastic_membership_noise=0.0, save_dir=output,
    )


def _inventory_args(method: str, seed: int, output: Path, scale: str, *, n_agg: int = 12) -> Namespace:
    from inventory_large_q_speed import make_base_args
    base_method = {"ts": "ts", "lp": "dfl", "qp": "qp", "sketch": "sketch"}.get(method, "sa")
    args = make_base_args(
        tier_name=scale, method=base_method, root=output.parents[3], seed=seed,
        n_samples=10 if scale == "basic" else 16, n_epochs=50, pretrain_epochs=30,
    )
    args.save_dir = output
    args.n_agg = -1 if method in {"ts", "lp", "qp", "sketch"} else int(n_agg)
    args.algo = {"ts": "twostage", "lp": "dfl", "qp": "dfl_qpth", "sketch": "dfl_sketch"}.get(method, "dfl")
    args.aug_lag_rho = {"ts": 10000.0, "lp": 10.0, "qp": 50.0, "sketch": 0.1}.get(method, 0.1)
    # The server has the structured QP backend; preserve the same task/model
    # while avoiding the dense qpth KKT allocation that is not feasible locally.
    if method == "qp" and os.environ.get("MARKOV_PROTOCOL_STRUCTURED_QP") == "1":
        args.qp_backend = "structured"
        args.structured_qp_tol = 1e-7
        args.structured_qp_max_iter = 500
    args.loss_weight = 0.0 if method == "ts" else 1.0
    if method == "sa":
        # Match the verified Inventory reproduction path.  This is the
        # ordinary-SA control that Markov variants must share, rather than
        # the later large-q DFL aggregation shortcut.
        args.algo = "dfl_sketch"
        args.n_sketches = 8
        args.sketch_ratio = 0.2
        args.sketch_mode = "gaussian"
        args.aug_lag_rho = 0.1
        args.loss_weight = 1.0
        if scale == "basic":
            # Exact fixed reproduction path: evaluate the lifted policy on
            # the original MDP, as in inventory_train_repro.py.
            args.use_aggregated_eval_policy = False
            args.use_cached_eval_oracle = False
            args.skip_initial_reward_eval = False
            args.use_aggregated_train_reward_eval = False
        # Medium keeps make_base_args' aggregated-policy evaluation.  The
        # original-state per-test LP is retained only as a one-seed resource
        # diagnostic, not used for the ten-seed Markov comparison.
        else:
            args.use_aggregated_train_reward_eval = True
            # Avoid the cached-data helper's full 16-sample batch, which
            # retains multiple diagnostic autograd graphs at this scale.
            # Eight remains a meaningful vectorized training batch.
            args.train_batch_size = 8
    args.sa_tau = 0.5
    args.agg_lr_ratio = 0.01
    # The fixed Inventory reproduction path warms up the predictor before
    # updating memberships.  Keeping that schedule is necessary for a fair
    # Markov versus ordinary-SA control.
    args.agg_warmup_epochs = 30
    args.checkpoint_every = 50
    return args


def _make_cliff_data(args: Namespace, stack):
    CliffDataConfig, prepare_cliff_data, _ = stack
    return prepare_cliff_data(CliffDataConfig(
        n_samples=args.n_samples, feat_dim=args.data_feat_dim, noise_std=args.noise_std, beta=args.beta,
        train_frac=args.train_frac, batch_size=args.batch_size, seed=args.seed, regime_set=args.regime_set,
        center_scale=args.center_scale, ambiguous_pairs=args.ambiguous_pairs, ambiguous_gap=args.ambiguous_gap,
        cliff_penalty=args.cliff_penalty, regime_weights=args.regime_weights, test_regime_weights=args.test_regime_weights,
    ))


def _make_inventory_data(args: Namespace, root: Path):
    from inventory_large_q_speed import prepare_cached_dataset
    return prepare_cached_dataset(root=root, tier_name="basic" if args.s_max == 30 else "medium", method="protocol", args=args, force_oracle=False, oracle_workers=4)


def _validation_loader(train_loader: DataLoader, seed: int) -> tuple[DataLoader, DataLoader]:
    size = len(train_loader.dataset)
    if size < 4:
        return train_loader, train_loader
    n_val = max(1, size // 4)
    train_set, val_set = random_split(train_loader.dataset, [size - n_val, n_val], generator=torch.Generator().manual_seed(seed + 71))
    batch_size = min(train_loader.batch_size or len(train_set), len(train_set))
    return DataLoader(train_set, batch_size=batch_size, shuffle=True), DataLoader(val_set, batch_size=min(batch_size, len(val_set)), shuffle=False)


def _geometry_from_agg(
    task: str,
    agg: ProtocolMarkovSoftAgg | SoftAggDense,
    data_loader: DataLoader,
    args: Namespace,
    spec: MarkovSpec,
    n_agg: int,
    membership_movement: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute geometry on one freshly generated held-out MDP, never an old checkpoint."""
    sample = next(iter(data_loader))
    if task == "cliff":
        P, r = sample[3][:1], sample[4][:1]
        gamma = torch.zeros(1, P.shape[1]); gamma[:, 0] = 1.0
    else:
        from mdp_lp.inventory_elasticity import create_inventory_from_lambda
        lam, price = sample[2][:1], sample[3][:1]
        mdp = create_inventory_from_lambda(lam, args.b_max, args.s_max, args.q_max, args.foc, args.voc, args.hc, args.bc, args.beta, price)
        P, r = mdp.P[:1], mdp.r[:1]
        gamma = mdp.build_gamma(1, args.b_max + args.s_max + 1, torch.device("cpu"))
        gamma = gamma / gamma.sum(dim=1, keepdim=True).clamp_min(1e-8)
    with torch.no_grad():
        _, _, _, info = agg(P, r, gamma, tau=float(getattr(args, "sa_tau", spec.tau)))
    W = info["W"][0].detach().cpu().numpy()
    sig = compute_markov_signature(P, r, max_depth=spec.depth,
                                   word_actions=choose_even_actions(P.shape[2], spec.n_word_actions),
                                   terminal_actions=choose_even_actions(P.shape[2], spec.n_terminal_actions))[0].detach().cpu().numpy()
    sig = (sig - sig.mean(0, keepdims=True)) / np.maximum(sig.std(0, keepdims=True), 1e-8)
    # Pairwise upper triangles retain the exact state pairing for Spearman.
    sig_dist = np.sqrt(((sig[:, None, :] - sig[None, :, :]) ** 2).sum(-1))
    w_dist = np.sqrt(((W[:, None, :] - W[None, :, :]) ** 2).sum(-1))
    upper = np.triu_indices(W.shape[0], k=1)
    corr = spearmanr(sig_dist[upper], w_dist[upper]).statistic
    labels_w = W.argmax(axis=1)
    km = KMeans(n_clusters=n_agg, n_init=20, random_state=0).fit_predict(sig)
    entropy = -(W * np.log(np.maximum(W, 1e-12))).sum(1).mean() / math.log(n_agg)
    mass = W.sum(0)
    centroids = (W.T @ sig) / np.maximum(mass[:, None], 1e-12)
    soft_cluster_signature_variance = float(
        (W * ((sig[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)).sum() / max(W.sum(), 1e-12)
    )
    adjacent_membership_discontinuity = float(np.abs(W[1:] - W[:-1]).sum(axis=-1).mean()) if W.shape[0] > 1 else 0.0
    # Markov neighbours are decision-relevant only if their greedy true reward action agrees.
    oracle_action = r[0].argmax(dim=-1).detach().cpu().numpy()
    np.fill_diagonal(sig_dist, np.inf)
    nearest = sig_dist.argmin(1)
    markov_agree = float((oracle_action == oracle_action[nearest]).mean())
    spatial = np.arange(W.shape[0] - 1)
    spatial_agree = float((oracle_action[:-1] == oracle_action[1:]).mean()) if spatial.size else float("nan")
    rng = np.random.default_rng(0)
    random_agree = float((oracle_action == oracle_action[rng.permutation(W.shape[0])]).mean())
    return {
        "spearman_signature_membership_distance": float(corr),
        "ari_hard_w_vs_signature_kmeans": float(adjusted_rand_score(labels_w, km)),
        "nmi_hard_w_vs_signature_kmeans": float(normalized_mutual_info_score(labels_w, km)),
        "normalized_membership_entropy": float(entropy),
        "cluster_mass_cv": float(mass.std() / max(mass.mean(), 1e-12)),
        "soft_cluster_signature_variance": soft_cluster_signature_variance,
        "adjacent_membership_discontinuity_l1": adjacent_membership_discontinuity,
        "markov_neighbor_action_agreement": markov_agree,
        "spatial_neighbor_action_agreement": spatial_agree,
        "random_action_agreement": random_agree,
        "membership_movement": membership_movement or {},
        "W": W.tolist(), "signature_kmeans_order": np.argsort(km).tolist(),
    }


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep the metric and convergence fields required by the notebook only."""
    wanted = {
        "train_reward_per_epoch", "train_pred_loss_per_epoch", "train_dfl_loss_per_epoch",
        "train_twostage_loss_per_epoch", "train_time", "solver_time_sec_per_epoch",
        "sa_time_sec_per_epoch", "backward_time_sec_per_epoch", "test_mean_pred_reward",
        "test_mean_oracle_reward", "test_mean_regret", "test_mean_regret_hinge", "test_slip_ce",
        "train_optimal_reward", "algo",
    }
    return {key: _jsonable(value) for key, value in result.items() if key in wanted}


def _numerical_events(log_path: Path) -> dict[str, int]:
    """Count explicit numerical warnings rather than inferring failures from regret."""
    text = log_path.read_text(encoding="utf-8", errors="replace").lower() if log_path.exists() else ""
    return {
        "nan_or_inf_mentions": text.count("nan") + text.count("inf"),
        "gradient_clip_mentions": text.count("clip"),
        "rejection_mentions": text.count("reject"),
    }


@contextmanager
def _capture_gradient_clipping():
    """Observe the true pre-clipping global norm used by the train loop."""
    original = torch.nn.utils.clip_grad_norm_
    history: list[dict[str, float | bool]] = []

    def wrapped(parameters, max_norm, *args, **kwargs):
        norm = original(parameters, max_norm, *args, **kwargs)
        value = float(norm.detach().cpu()) if isinstance(norm, torch.Tensor) else float(norm)
        history.append({"pre_clip_norm": value, "max_norm": float(max_norm), "was_clipped": bool(value > float(max_norm))})
        return norm

    torch.nn.utils.clip_grad_norm_ = wrapped
    try:
        yield history
    finally:
        torch.nn.utils.clip_grad_norm_ = original


def _clip_summary(history: list[dict[str, float | bool]]) -> dict[str, Any]:
    values = [float(row["pre_clip_norm"]) for row in history]
    return {
        "n_steps": len(history),
        "clipped_steps": int(sum(bool(row["was_clipped"]) for row in history)),
        "pre_clip_norm": ProtocolMarkovSoftAgg._distribution(values),
    }


def run_worker(task: str, scale: str, method: str, seed_index: int, root: Path, *, screen: bool = False, spec: MarkovSpec | None = None, force: bool = False, data_root: Path | None = None) -> dict[str, Any]:
    """Run one fresh seed.  Errors are recorded as observations, not hidden."""
    # Each worker is isolated. Cap intra-op parallelism to prevent dense
    # medium-Cliff construction from oversubscribing the server CPU.
    cpu_threads = int(os.environ.get("MARKOV_PROTOCOL_CPU_THREADS", "8"))
    torch.set_num_threads(max(1, cpu_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    raw_seed = (INVENTORY_RAW_SEEDS if task == "inventory" else CLIFF_RAW_SEEDS)[seed_index]
    path_name = next((path for path in FINAL_PATHS if method == path or method.startswith(f"{path}_m")), None)
    method_n_agg = _method_n_agg(method)
    if spec is not None and method_n_agg != spec.n_agg:
        spec = replace(spec, n_agg=method_n_agg)
    n_agg = spec.n_agg if spec else method_n_agg
    output = root / task / scale / method / f"seed_{seed_index}"
    data_root = data_root or (root / "data")
    result_path = output / "record.json"
    # A screen label is an output identifier; training still uses the learned
    # SA path with the candidate Markov specification.
    path = "markov_scaled" if screen else path_name
    if path and spec is None:
        raise ValueError("Markov method requires a selected specification")
    train_method = "sa" if method.startswith("sa_") or path else method
    args = _inventory_args(train_method, raw_seed, output, scale, n_agg=n_agg) if task == "inventory" else _cliff_args(train_method, raw_seed, output, scale, n_agg=n_agg)
    config = _protocol_config(
        task=task, scale=scale, method=method, raw_seed=raw_seed, screen=screen,
        spec=spec, n_agg=n_agg, args=args,
    )
    fingerprint = _protocol_fingerprint(config)
    if result_path.exists() and not force:
        record = json.loads(result_path.read_text(encoding="utf-8"))
        if record.get("status") == "ok" and record.get("protocol_fingerprint") == fingerprint:
            # Preserve the completed/failed outcome for selection and summaries;
            # cache provenance is a separate field, not a third execution status.
            record["cache_hit"] = True
            return record
    if force and output.exists():
        shutil.rmtree(output)
    elif output.exists():
        # The existing file belongs to a different protocol revision.  Replace
        # only this worker directory, never a neighbouring seed/method result.
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    registry: list[ProtocolMarkovSoftAgg] = []
    normal_registry: list[SoftAggDense] = []
    started = time.perf_counter()
    train_loader = test_loader = None
    execution_metadata: dict[str, Any] = {}
    try:
        with ResourceMonitor(0.1) as profile:
            with (output / "run.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                with _capture_gradient_clipping() as clipping_history:
                    if task == "inventory":
                        from mdp_lp import train_elasticity as train_mod
                        # State aggregation must keep its aggregation solver
                        # path at every scale.  Replacing it with the large-q
                        # SciPy full-LP backend silently solves the original
                        # MDP inside each SA step and defeats the purpose of
                        # this comparison.
                        if train_method != "sa":
                            from inventory_large_q_smoke import install_scipy_lp_backend
                            install_scipy_lp_backend(generic_constraints=(train_method == "sketch"))
                        train_loader, test_loader, gt, cache_info = _make_inventory_data(args, data_root)
                        if screen:
                            train_loader, evaluation_loader = _validation_loader(train_loader, raw_seed)
                        else:
                            evaluation_loader = DataLoader(test_loader.dataset, batch_size=1, shuffle=False)
                        execution_metadata = {
                            "train_batch_size": train_loader.batch_size,
                            "evaluation_batch_size": evaluation_loader.batch_size,
                            "train_workers": train_loader.num_workers,
                            "evaluation_workers": evaluation_loader.num_workers,
                            "solver_backend": getattr(args, "qp_backend", "scipy_lp_or_repo_default"),
                        }
                        if method.startswith("sa_m"):
                            with _capture_normal_aggregator(task, normal_registry):
                                result = train_mod.train_eval_elasticity(args, train_loader, evaluation_loader, gt)
                        else:
                            with _patch_aggregator(task, spec, path, registry):
                                result = train_mod.train_eval_elasticity(args, train_loader, evaluation_loader, gt)
                        extra = {"oracle_cache_wall_sec": cache_info["oracle_wall_sec"]}
                    else:
                        stack = _cliff_setup(scale)
                        _, _, train_mod = stack
                        train_loader, test_loader, meta = _make_cliff_data(args, stack)
                        if screen:
                            train_loader, evaluation_loader = _validation_loader(train_loader, raw_seed)
                        else:
                            # Keep the training batch at 32.  Evaluation has an
                            # independently controlled, moderate solve batch.
                            # This avoids the unstable 32-way 480-state LU while
                            # retaining practical evaluation throughput.
                            eval_batch_size = (
                                args.batch_size if scale == "basic"
                                else max(1, int(os.environ.get("MARKOV_PROTOCOL_CLIFF_EVAL_BATCH_SIZE", "1")))
                            )
                            evaluation_loader = DataLoader(
                                test_loader.dataset,
                                batch_size=eval_batch_size,
                                shuffle=False,
                            )
                        execution_metadata = {
                            "train_batch_size": train_loader.batch_size,
                            "evaluation_batch_size": evaluation_loader.batch_size,
                            "train_workers": train_loader.num_workers,
                            "evaluation_workers": evaluation_loader.num_workers,
                            "solver_backend": getattr(args, "qp_backend", "repo_default"),
                        }
                        # Suppress only per-run plot side effects; the result and training are unchanged.
                        import types
                        plot_mod = types.ModuleType("mdp_cw.plot_cliff_run")
                        plot_mod.make_all_plots = lambda _: None
                        sys.modules["mdp_cw.plot_cliff_run"] = plot_mod
                        if method.startswith("sa_m"):
                            with _capture_normal_aggregator(task, normal_registry):
                                result = train_mod.train_eval_cliff(args, train_loader, evaluation_loader, meta)
                        else:
                            with _patch_aggregator(task, spec, path, registry):
                                result = train_mod.train_eval_cliff(args, train_loader, evaluation_loader, meta)
                        extra = {}
        resource = profile.result().to_dict()
        execution_metadata["observed_cuda_allocated_peak_bytes"] = resource.get("gpu_allocated_peak_bytes", 0)
        execution_metadata["observed_cuda_reserved_peak_bytes"] = resource.get("gpu_reserved_peak_bytes", 0)
        gradient = registry[-1].gradient_summary() if registry else {}
        update = registry[-1].parameter_update_summary() if registry else {}
        trainable = registry[-1].trainable_definition() if registry else {
            "base_membership": ["SoftAggDense base membership parameters"],
            "markov_membership": [],
            "signature_to_predictor_gradient": ["not applicable"],
        }
        geometry = None
        # Screen workers are validation-only selection evidence.  They do not
        # compute any held-out geometry, so downstream final geometry cannot
        # accidentally contain a test-set probe from a screen run.
        if not screen and method.startswith("sa_m") and normal_registry:
            normal = normal_registry[-1]
            movement = normal.membership_movement(float(args.sa_tau))
            geometry = _geometry_from_agg(
                task, normal, test_loader, args, MarkovSpec(n_agg=n_agg, tau=args.sa_tau), n_agg, movement,
            )
            update = normal.parameter_update_summary()
        elif not screen and registry:
            markov = registry[-1]
            geometry = _geometry_from_agg(
                task, markov, test_loader, args, spec or MarkovSpec(n_agg=n_agg, tau=args.sa_tau), n_agg,
                markov.membership_movement(float(args.sa_tau)),
            )
        if registry:
            torch.save({"state_dict": registry[-1].state_dict(), "spec": asdict(spec) if spec else None, "path": path}, output / "aggregation_state.pt")
        elif normal_registry:
            # Keep the trained ordinary-SA aggregator as well.  Its predictor
            # checkpoints alone cannot reconstruct the final co-assignment
            # matrix needed for paired Markov geometry measurements.
            torch.save({"state_dict": normal_registry[-1].state_dict(), "spec": None, "path": "ordinary_sa"}, output / "aggregation_state.pt")
        record = {
            "status": "ok", "task": task, "scale": scale, "method": method, "screen": screen,
            "evaluation_split": "validation" if screen else "held_out_test",
            "seed_index": seed_index, "raw_seed": raw_seed,
            "n_states": (args.b_max + args.s_max + 1) if task == "inventory" else _cliff_state_count(scale),
            "n_actions": (args.q_max + 1) if task == "inventory" else 4, "n_agg": n_agg,
            "spec": asdict(spec) if spec else None, "result": _compact_result(result), "resource": resource,
            "gradient": gradient, "parameter_update": update, "trainable_definition": trainable,
            "geometry": geometry, "numerical_events": _numerical_events(output / "run.log"),
            "gradient_clipping": _clip_summary(clipping_history),
            "execution_metadata": execution_metadata,
            "protocol_config": config, "protocol_fingerprint": fingerprint,
            "wall_time_sec": time.perf_counter() - started, **extra,
        }
    except Exception as exc:
        record = {
            "status": "failed", "task": task, "scale": scale, "method": method, "screen": screen,
            "seed_index": seed_index, "raw_seed": raw_seed, "n_agg": n_agg,
            "spec": asdict(spec) if spec else None, "wall_time_sec": time.perf_counter() - started,
            "protocol_config": config, "protocol_fingerprint": fingerprint,
            "error": repr(exc), "traceback": traceback.format_exc(),
        }
    result_path.write_text(json.dumps(_jsonable(record), indent=2), encoding="utf-8")
    return record


def _mean_se(values: Iterable[float]) -> tuple[float, float]:
    v = np.asarray(list(values), dtype=float)
    if not v.size:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0


def summarize(root: Path) -> dict[str, Any]:
    rows = []
    for path in root.glob("*/*/*/seed_*/record.json"):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    groups: dict[tuple[str, str, str, bool], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["task"], row["scale"], row["method"], bool(row.get("screen"))), []).append(row)
    summaries = []
    for (task, scale, method, screen), group in sorted(groups.items()):
        ok = [r for r in group if r["status"] == "ok"]
        metric = lambda name: [float(r["result"].get(name, np.nan)) for r in ok if name in r.get("result", {})]
        regret, regret_se = _mean_se(metric("test_mean_regret"))
        reward, reward_se = _mean_se(metric("test_mean_pred_reward"))
        oracle, oracle_se = _mean_se(metric("test_mean_oracle_reward"))
        wall, wall_se = _mean_se([float(r.get("wall_time_sec", np.nan)) for r in ok])
        solver, solver_se = _mean_se([sum(r["result"].get("solver_time_sec_per_epoch", [])) for r in ok])
        agg, agg_se = _mean_se([sum(r["result"].get("sa_time_sec_per_epoch", [])) for r in ok])
        summaries.append({
            "task": task, "scale": scale, "method": method, "screen": screen,
            "completed": len(ok), "attempted": len(group), "completion_rate": len(ok) / max(1, len(group)),
            "regret_mean": regret, "regret_se": regret_se, "reward_mean": reward, "reward_se": reward_se,
            "oracle_mean": oracle, "oracle_se": oracle_se, "wall_mean_sec": wall, "wall_se_sec": wall_se,
            "solver_mean_sec": solver, "solver_se_sec": solver_se, "aggregation_mean_sec": agg, "aggregation_se_sec": agg_se,
            "peak_rss_mean_bytes": _mean_se([r.get("resource", {}).get("rss_peak_bytes", np.nan) for r in ok])[0],
            "peak_cuda_allocated_mean_bytes": _mean_se([r.get("resource", {}).get("gpu_allocated_peak_bytes", np.nan) for r in ok])[0],
            "failure_types": sorted({r.get("error", "")[:180] for r in group if r["status"] != "ok"}),
        })
    payload = {"protocol": "markov_notebook_execution_plan", "summaries": summaries, "records": rows}
    (root / "summary.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    return payload


def run_protocol(root: Path, *, force: bool, phases: list[str]) -> dict[str, Any]:
    """Run in protocol order.  Each worker is a fresh Python process.

    In particular, ``mdp_cw.cliff_mdp`` uses module-level grid constants.  A
    fresh worker is therefore required to make basic and medium Cliff data
    genuinely distinct rather than merely relabelled runs.
    """
    root.mkdir(parents=True, exist_ok=True)
    final_seeds = range(10)
    # The predeclared Path-3 representation screen is deliberately small.
    # It uses only validation data; the held-out test set is reserved for the
    # subsequent ten-seed final comparison.
    screen_seeds = range(3)
    scales = ACTIVE_CELLS

    def invoke(task: str, scale: str, method: str, seed: int, *, screen: bool = False, spec: MarkovSpec | None = None, screen_index: int | None = None) -> dict[str, Any]:
        cmd = [
            sys.executable, str(Path(__file__).resolve()), "--root", str(root), "--task", task,
            "--scale", scale, "--method", method, "--seed-index", str(seed),
        ]
        if force:
            cmd.append("--force")
        if screen:
            cmd.append("--screen")
        if screen_index is not None:
            cmd.extend(["--screen-spec-index", str(screen_index)])
        elif spec is not None:
            spec_path = root / "selected_specs" / f"{task}_{scale}.json"
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")
            cmd.extend(["--spec-json", str(spec_path)])
        record_path = root / task / scale / method / f"seed_{seed}" / "record.json"
        timeout_env = "MARKOV_PROTOCOL_SKETCH_TIMEOUT_SEC" if method == "sketch" else "MARKOV_PROTOCOL_WORKER_TIMEOUT_SEC"
        timeout_sec = int(os.environ.get(timeout_env, os.environ.get("MARKOV_PROTOCOL_WORKER_TIMEOUT_SEC", "7200")))
        try:
            completed = subprocess.run(
                cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps({
                "status": "failed", "task": task, "scale": scale, "method": method, "screen": screen,
                "evaluation_split": "validation" if screen else "held_out_test", "seed_index": seed,
                "raw_seed": (INVENTORY_RAW_SEEDS if task == "inventory" else CLIFF_RAW_SEEDS)[seed],
                "n_agg": _method_n_agg(method),
                "spec": asdict(spec) if spec else None,
                "error": f"worker_timeout_after_{timeout_sec}_seconds",
                "worker_stdout_tail": _jsonable(exc.stdout[-3000:] if isinstance(exc.stdout, str) else ""),
            }, indent=2), encoding="utf-8")
        if not record_path.exists():
            raise RuntimeError(f"Worker did not write {record_path}; output:\n{completed.stdout[-3000:]}")
        return json.loads(record_path.read_text(encoding="utf-8"))

    selected: MarkovSpec | None = None
    if "screen" in phases:
        # One representation is selected globally from validation-only screens.
        # Every score is keyed by its named task/scale cell.  A missing cell
        # therefore cannot shift a later score into the wrong rank position.
        screen_cells: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
        for task, scale in scales:
            cell = (task, scale)
            screen_cells[cell] = {}
            for index, candidate in enumerate(SCREEN_SPECS):
                label = f"screen_{index}"
                candidates = []
                for seed in screen_seeds:
                    row = invoke(task, scale, label, seed, screen=True, spec=candidate, screen_index=index)
                    candidates.append(row)
                ok = [r for r in candidates if r["status"] == "ok" and np.isfinite(r.get("result", {}).get("test_mean_regret", np.nan))]
                screen_cells[cell][index] = {
                    "candidate_index": index,
                    "spec": asdict(candidate),
                    "attempted": len(candidates),
                    "completed": len(ok),
                    "status": "complete" if len(ok) == len(screen_seeds) else "incomplete",
                    "validation_regret_mean": float(np.mean([float(r["result"]["test_mean_regret"]) for r in ok])) if ok else None,
                    "failed_seeds": [int(r["seed_index"]) for r in candidates if r["status"] != "ok"],
                }
        # Each cell contributes an equal within-cell rank.  A candidate is
        # globally eligible only if every active cell completed all screen seeds.
        global_ranks: dict[int, list[int]] = {index: [] for index in range(len(SCREEN_SPECS))}
        for cell in scales:
            complete = [
                (index, values)
                for index, values in screen_cells[cell].items()
                if values["status"] == "complete" and values["validation_regret_mean"] is not None
            ]
            for rank, (index, values) in enumerate(sorted(complete, key=lambda item: float(item[1]["validation_regret_mean"]))):
                values["within_cell_rank"] = rank + 1
                global_ranks[index].append(rank + 1)
            for index, values in screen_cells[cell].items():
                values.setdefault("within_cell_rank", None)
        candidate_summary = []
        for index, candidate in enumerate(SCREEN_SPECS):
            eligible = len(global_ranks[index]) == len(scales)
            candidate_summary.append({
                "candidate_index": index,
                "spec": asdict(candidate),
                "completed_cells": len(global_ranks[index]),
                "global_mean_rank": float(np.mean(global_ranks[index])) if global_ranks[index] else None,
                "eligible_for_final": eligible,
            })
        viable = [row for row in candidate_summary if row["eligible_for_final"]]
        if viable:
            chosen = min(viable, key=lambda row: (float(row["global_mean_rank"]), int(row["candidate_index"])))
            selected = SCREEN_SPECS[int(chosen["candidate_index"])]
        provenance = {
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "selection_split": "validation_only",
            "screen_seed_indices": list(screen_seeds),
            "task_scale_cells": [{"task": task, "scale": scale} for task, scale in scales],
            "candidate_cells": {
                f"{task}/{scale}": [screen_cells[(task, scale)][index] for index in range(len(SCREEN_SPECS))]
                for task, scale in scales
            },
            "candidate_summary": candidate_summary,
            "selected_candidate_index": None if selected is None else next(index for index, spec in enumerate(SCREEN_SPECS) if spec == selected),
            "selected_specification": asdict(selected) if selected is not None else None,
            "final_held_out_test_used_for_selection": False,
        }
        (root / "selection_provenance.json").write_text(json.dumps(_jsonable(provenance), indent=2), encoding="utf-8")
    selection_path = root / "selected_representation.json"
    if selected is not None:
        selection_path.write_text(json.dumps({
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "global_validation_selected_spec": asdict(selected),
            "selection_provenance": "selection_provenance.json",
        }, indent=2), encoding="utf-8")
    elif selection_path.exists():
        saved = json.loads(selection_path.read_text(encoding="utf-8"))
        selected = MarkovSpec(**saved["global_validation_selected_spec"])
    if "final" in phases:
        for task, scale in scales:
            if selected is None:
                raise RuntimeError("No frozen global Path-3 selection; run --phases screen first.")
            for n_agg in MARKOV_SIZES[(task, scale)]:
                candidate = replace(selected, n_agg=n_agg)
                for path in FINAL_PATHS:
                    for seed in final_seeds:
                        invoke(task, scale, f"{path}_m{n_agg}", seed, spec=candidate)
    return summarize(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=("inventory", "cliff"))
    parser.add_argument("--scale", choices=("basic", "medium"))
    parser.add_argument("--method")
    parser.add_argument("--seed-index", type=int)
    parser.add_argument("--spec-json", type=Path)
    parser.add_argument("--screen-spec-index", type=int, choices=range(len(SCREEN_SPECS)))
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--protocol", action="store_true")
    parser.add_argument("--phases", nargs="+", default=["screen", "final"], choices=("screen", "final"))
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(summarize(args.root)["summaries"], indent=2))
        return 0
    if args.protocol:
        print(json.dumps(run_protocol(args.root, force=args.force, phases=args.phases)["summaries"], indent=2))
        return 0
    if not (args.task and args.scale and args.method is not None and args.seed_index is not None):
        parser.error("worker mode requires --task --scale --method --seed-index")
    spec = MarkovSpec(**json.loads(args.spec_json.read_text(encoding="utf-8"))) if args.spec_json else None
    if args.screen_spec_index is not None:
        spec = SCREEN_SPECS[args.screen_spec_index]
    record = run_worker(args.task, args.scale, args.method, args.seed_index, args.root, screen=args.screen, spec=spec, force=args.force)
    print(json.dumps({
        "status": record["status"], "task": record["task"], "scale": record["scale"],
        "method": record["method"], "seed_index": record["seed_index"],
        "regret": record.get("result", {}).get("test_mean_regret"),
        "wall_time_sec": record.get("wall_time_sec"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
