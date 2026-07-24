from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

REPO = Path(__file__).resolve().parents[3]
AGG_DIR = Path(__file__).resolve().parent
if str(AGG_DIR) not in sys.path:
    sys.path.insert(0, str(AGG_DIR))

from markov_signature_aggregation import (  # noqa: E402
    MarkovSignatureConfig,
    MarkovSignatureSoftAgg,
    StochasticMarkovSignatureSoftAgg,
    aggregate_with_membership,
)
from mdp_lp.soft_state_aggregation import SoftAggDense  # noqa: E402


def build_markov_signature_agg(
    *,
    n_states: int,
    n_actions: int,
    n_clusters: int,
    signature_depth: int = 1,
    n_word_actions: int | None = None,
    n_terminal_actions: int | None = None,
    stochastic: bool = False,
    n_membership_samples: int = 2,
    noise_std: float = 0.15,
    seed: int | None = None,
) -> nn.Module:
    config = MarkovSignatureConfig(
        n_states=n_states,
        n_actions=n_actions,
        n_clusters=n_clusters,
        signature_depth=signature_depth,
        n_word_actions=n_word_actions,
        n_terminal_actions=n_terminal_actions,
        encoder_hidden=32,
        embed_dim=16,
        use_raw_state_index=True,
        normalize_signature=True,
        seed=seed,
    )
    if stochastic:
        return StochasticMarkovSignatureSoftAgg(
            config,
            n_membership_samples=n_membership_samples,
            noise_std=noise_std,
        )
    return MarkovSignatureSoftAgg(config)


class StochasticSoftAggDense(nn.Module):
    """Stochastic membership sampler around the existing SoftAggDense logits.

    The wrapper first computes the ordinary soft aggregation membership W, then
    samples nearby memberships by perturbing log W. This keeps the original
    learned membership mechanism but makes the LP layer average over a small
    local ensemble of reduced MDPs.
    """

    def __init__(
        self,
        *,
        n_states: int,
        n_actions: int,
        n_clusters: int,
        r_embed: int = 8,
        n_membership_samples: int = 2,
        noise_std: float = 0.15,
    ):
        super().__init__()
        self.base = SoftAggDense(
            n_states=n_states,
            n_actions=n_actions,
            n_clusters=n_clusters,
            r_embed=r_embed,
        )
        self.n_membership_samples = max(1, int(n_membership_samples))
        self.noise_std = float(noise_std)

    def forward(
        self,
        P: torch.Tensor,
        r: torch.Tensor,
        gamma: torch.Tensor,
        tau: float = 0.5,
    ):
        start = time.time()
        P_base, r_base, gamma_base, info = self.base(P, r, gamma, tau=tau)
        W_base = info["W"]
        # Eq. (61) perturbs the pre-softmax logits, rather than log(W) after
        # temperature has already been applied.  Mirror SoftAggDense's
        # original membership map exactly so the deterministic sample remains
        # its sigma=0 special case.
        if self.base.fixed_W:
            logits = torch.log(W_base.clamp_min(1e-8)) * float(tau)
        else:
            embedding = self.base.dest_embed.weight
            phi = torch.einsum("bsaj,jr->bsar", P, embedding)
            logits = self.base.to_logits(phi.flatten(start_dim=2)) + self.base.r_bias(r)
        samples = []
        if self.training and self.noise_std > 0.0:
            for _ in range(self.n_membership_samples):
                W = F.softmax(
                    (logits + torch.randn_like(logits) * self.noise_std) / float(tau),
                    dim=-1,
                )
                P_agg, r_agg, gamma_agg, mass = aggregate_with_membership(P, r, gamma, W)
                samples.append((P_agg, r_agg, gamma_agg, {"W": W, "mass": mass, "logits": logits}))
        # Do not average reduced coefficients here.  The training layer must
        # solve and lift every sampled occupancy before averaging it (Algorithm
        # 1, Eqs. 61--65).  Returning an averaged P/r/gamma with W_base would
        # mix incompatible memberships and silently bypass that mechanism.
        merged = dict(info)
        merged.update({"W": W_base, "logits": logits, "runtime_sec": time.time() - start})
        # Presence of this key is the training-loop signal that samplewise LP
        # solves are required.  In evaluation mode the deterministic layer is
        # used, so omit it rather than returning an empty list.
        if samples:
            merged["samples"] = samples
        return P_base, r_base, gamma_base, merged


class CoefficientSmoothedSoftAggDense(nn.Module):
    """Coefficient-smoothed variant of the historical stochastic Cliff path.

    This is intentionally *not* the sampled-solve-lift estimator in Algorithm
    1.  It samples nearby memberships, aggregates each sampled MDP, averages
    the reduced coefficients, then solves that one smoothed reduced LP.  The
    ``lift_membership`` option makes the historical base-W lift explicit so it
    can be tested rather than silently conflated with the pathwise method.

    It is retained as a separately named surrogate because prior Cliff runs
    indicated a useful signal.  It must earn that result again under a paired
    rerun before being treated as an empirical method.
    """

    def __init__(
        self,
        *,
        n_states: int,
        n_actions: int,
        n_clusters: int,
        r_embed: int = 8,
        n_membership_samples: int = 2,
        noise_std: float = 0.15,
        noise_parameterization: str = "logw",
        lift_membership: str = "base",
    ):
        super().__init__()
        self.base = SoftAggDense(
            n_states=n_states,
            n_actions=n_actions,
            n_clusters=n_clusters,
            r_embed=r_embed,
        )
        self.n_membership_samples = max(1, int(n_membership_samples))
        self.noise_std = float(noise_std)
        if noise_parameterization not in {"logw", "logit"}:
            raise ValueError("noise_parameterization must be 'logw' or 'logit'")
        if lift_membership not in {"base", "mean"}:
            raise ValueError("lift_membership must be 'base' or 'mean'")
        self.noise_parameterization = noise_parameterization
        self.lift_membership = lift_membership

    def _logits(self, P: torch.Tensor, r: torch.Tensor, W_base: torch.Tensor, tau: float) -> torch.Tensor:
        if self.base.fixed_W:
            return torch.log(W_base.clamp_min(1e-8)) * float(tau)
        embedding = self.base.dest_embed.weight
        phi = torch.einsum("bsaj,jr->bsar", P, embedding)
        return self.base.to_logits(phi.flatten(start_dim=2)) + self.base.r_bias(r)

    def forward(
        self,
        P: torch.Tensor,
        r: torch.Tensor,
        gamma: torch.Tensor,
        tau: float = 0.5,
    ):
        start = time.time()
        P_base, r_base, gamma_base, info = self.base(P, r, gamma, tau=tau)
        W_base = info["W"]
        logits = self._logits(P, r, W_base, tau)
        if not self.training or self.noise_std <= 0.0:
            merged = dict(info)
            merged.update({"W": W_base, "logits": logits, "runtime_sec": time.time() - start})
            return P_base, r_base, gamma_base, merged

        reduced = [(P_base, r_base, gamma_base)]
        memberships = [W_base]
        for _ in range(self.n_membership_samples):
            if self.noise_parameterization == "logw":
                # Historical implementation: perturb after the temperature map.
                W_k = F.softmax(torch.log(W_base.clamp_min(1e-8)) + torch.randn_like(W_base) * self.noise_std, dim=-1)
            else:
                # Eq. (61)-style logits, kept separate from the historical form.
                W_k = F.softmax((logits + torch.randn_like(logits) * self.noise_std) / float(tau), dim=-1)
            reduced.append(aggregate_with_membership(P, r, gamma, W_k)[:3])
            memberships.append(W_k)

        P_smooth = torch.stack([item[0] for item in reduced]).mean(dim=0)
        r_smooth = torch.stack([item[1] for item in reduced]).mean(dim=0)
        gamma_smooth = torch.stack([item[2] for item in reduced]).mean(dim=0)
        W_lift = W_base if self.lift_membership == "base" else torch.stack(memberships).mean(dim=0)
        merged = dict(info)
        merged.update({
            "W": W_lift,
            "base_W": W_base,
            "mean_W": torch.stack(memberships).mean(dim=0),
            "logits": logits,
            "coefficient_smoothing": True,
            "noise_parameterization": self.noise_parameterization,
            "lift_membership": self.lift_membership,
            "runtime_sec": time.time() - start,
        })
        # Intentionally no ``samples`` key: the training loop solves the one
        # coefficient-smoothed LP described above.
        return P_smooth, r_smooth, gamma_smooth, merged
