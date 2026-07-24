"""Anchored Markov-prototype membership smoke.

This is a repair of the direct-prototype smoke, not another additive-logit
variant.  The previous M=S sanity run showed prototype collapse during DFL:
an initially near-identity state representation lost cluster coverage and the
training reward collapsed.  Here prototypes remain learnable but are bounded
around their Markov/chi anchors:

    q_m = q_m^(0) + eps_q tanh(delta_m),
    h(s) = [g_L(s) + eps_h tanh(R(g_L(s))), c chi(s)],
    W(s,m) = softmax(-||h(s)-q_m||^2 / tau).

The fixed raw coordinate chi(s) is permitted by the brainstorm Eq. (58), and
the bounded deviations preserve coverage needed by a reduced occupancy model.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import markov_protocol_runner as base


SPEC = base.MarkovSpec(
    depth=2,
    n_word_actions=4,
    n_terminal_actions=4,
    hidden=24,
    embed=12,
    alpha=1.0,
    gradient_scale=1.0,
    tau=0.02,
    use_raw_state_index=True,
)


class AnchoredPrototypeAgg(base.ProtocolMarkovSoftAgg):
    signature_residual_scale = 0.05
    prototype_delta_scale = 0.05

    def __init__(
        self, *args: Any, spec: base.MarkovSpec, path: str, registry=None,
        state_coordinate_scale: float, **kwargs: Any,
    ):
        super().__init__(*args, spec=spec, path=path, registry=registry, **kwargs)
        if not spec.use_raw_state_index:
            raise ValueError("AnchoredPrototypeAgg requires an explicit chi(s) coordinate")
        self.state_coordinate_scale = float(state_coordinate_scale)
        self.signature_residual = nn.Linear(self.signature_dim - 1, self.signature_dim - 1, bias=False)
        nn.init.zeros_(self.signature_residual.weight)
        self.prototype_delta = nn.Parameter(torch.zeros(self.n_clusters, self.signature_dim))
        self.register_buffer("prototype_anchor", torch.zeros(self.n_clusters, self.signature_dim))
        self.register_buffer("prototypes_initialized", torch.tensor(False))
        self._initial_state = {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}

    def _representation(self, P: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        raw = self._signature(P, r)
        signature, coordinate = raw[..., :-1], raw[..., -1:]
        signature = signature + self.signature_residual_scale * torch.tanh(self.signature_residual(signature))
        return torch.cat([signature, self.state_coordinate_scale * coordinate], dim=-1)

    def _initialize_anchor(self, representation: torch.Tensor) -> None:
        if bool(self.prototypes_initialized):
            return
        with torch.no_grad():
            state_index = torch.linspace(
                0, representation.shape[1] - 1, steps=self.n_clusters, device=representation.device
            ).round().long()
            self.prototype_anchor.copy_(representation[0, state_index])
            self.prototypes_initialized.fill_(True)

    def compute_logits(self, P: torch.Tensor, r: torch.Tensor):
        representation = self._representation(P, r)
        self._initialize_anchor(representation)
        prototypes = self.prototype_anchor + self.prototype_delta_scale * torch.tanh(self.prototype_delta)
        logits = -(
            representation.unsqueeze(2) - prototypes.view(1, 1, self.n_clusters, -1)
        ).pow(2).mean(dim=-1)
        if torch.is_grad_enabled() and logits.requires_grad:
            logits.register_hook(
                lambda grad: self.logit_gradient_norms["markov_logits"].append(float(grad.norm().detach().cpu()))
            )
        return logits, torch.zeros_like(logits), logits

    def trainable_definition(self) -> dict[str, list[str]]:
        return {
            "base_membership": [],
            "markov_membership": ["prototype_delta", "signature_residual"],
            "prototype_constraint": [f"anchor + {self.prototype_delta_scale:g} tanh(delta)"],
            "fixed_chi": [f"raw state coordinate x{self.state_coordinate_scale:g}"],
            "membership_form": ["softmax(-mean((h(g_L, chi)-q_m)^2)/tau)"],
        }

    def optimizer_parameter_groups(self, base_lr: float):
        return [{"params": self.parameters(), "lr": float(base_lr) * 10.0}]


def make_aggregator(spec: base.MarkovSpec, path: str, state_coordinate_scale: float, registry):
    class ConfiguredAnchoredPrototype(AnchoredPrototypeAgg):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(
                *args, spec=spec, path=path, registry=registry,
                state_coordinate_scale=state_coordinate_scale, **kwargs,
            )
    return ConfiguredAnchoredPrototype


@contextmanager
def patched_aggregator(spec: base.MarkovSpec, state_coordinate_scale: float):
    old_make = base._make_protocol_agg
    old_args = base._inventory_args

    def inventory_args(*args: Any, **kwargs: Any):
        result = old_args(*args, **kwargs)
        result.sa_tau = float(spec.tau)
        result.agg_lr_ratio = 0.1
        return result

    base._make_protocol_agg = lambda _spec, path, registry: make_aggregator(spec, path, state_coordinate_scale, registry)
    base._inventory_args = inventory_args
    try:
        yield
    finally:
        base._make_protocol_agg = old_make
        base._inventory_args = old_args


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "notebooks" / "reproduction" / "markov_prototype_anchored_smoke")
    parser.add_argument("--data-root", type=Path, default=REPO / ".cache" / "markov_aggregation")
    parser.add_argument("--n-agg", type=int, default=36)
    parser.add_argument("--tau", type=float, default=SPEC.tau)
    parser.add_argument("--state-coordinate-scale", type=float, default=50.0)
    parser.add_argument("--path", choices=("markov_e2e", "markov_detach", "markov_scaled"), default="markov_e2e")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--include-ordinary-control",
        action="store_true",
        help="Train the matched ordinary-SA control with the same data, M, tau, and aggregation LR.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    spec = replace(SPEC, n_agg=int(args.n_agg), tau=float(args.tau))

    records = []
    with patched_aggregator(spec, args.state_coordinate_scale):
        if args.include_ordinary_control:
            for seed in args.seeds:
                records.append(base.run_worker(
                    "inventory", "basic", f"sa_m{spec.n_agg}", int(seed), args.root, spec=None, force=args.force, data_root=args.data_root
                ))
        for seed in args.seeds:
            records.append(base.run_worker(
                "inventory", "basic", f"{args.path}_m{spec.n_agg}", int(seed), args.root, spec=spec, force=args.force, data_root=args.data_root
            ))
    payload = {
        "kind": "anchored_markov_prototype_smoke", "spec": asdict(spec),
        "state_coordinate_scale": float(args.state_coordinate_scale), "seed_indices": list(args.seeds),
        "records": records,
    }
    (args.root / "smoke_summary.json").write_text(json.dumps(base._jsonable(payload), indent=2), encoding="utf-8")
    print(json.dumps({"completed": sum(r["status"] == "ok" for r in records), "attempted": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
