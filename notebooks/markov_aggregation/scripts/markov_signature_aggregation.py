from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def all_action_words(n_actions: int, max_depth: int, action_subset: Iterable[int] | None = None):
    """Return open-loop action words with length 0..max_depth."""
    if max_depth < 0:
        raise ValueError("max_depth must be nonnegative")
    actions = list(range(n_actions)) if action_subset is None else list(action_subset)
    words: list[tuple[int, ...]] = [()]
    for depth in range(1, max_depth + 1):
        words.extend(tuple(word) for word in itertools.product(actions, repeat=depth))
    return words


def choose_even_actions(n_actions: int, n_keep: int | None) -> list[int]:
    """Choose a deterministic action subset for compact signatures."""
    if n_keep is None or n_keep >= n_actions:
        return list(range(n_actions))
    if n_keep <= 0:
        raise ValueError("n_keep must be positive")
    idx = torch.linspace(0, n_actions - 1, n_keep).round().long().unique()
    return [int(x) for x in idx.tolist()]


def compute_markov_signature(
    P: torch.Tensor,
    r: torch.Tensor,
    *,
    max_depth: int = 1,
    word_actions: Iterable[int] | None = None,
    terminal_actions: Iterable[int] | None = None,
) -> torch.Tensor:
    """Compute finite-horizon open-loop reward signatures.

    signature[b,s,:] stacks E[r(s_l,b_action) | s_0=s, action word alpha]
    for all selected alpha with |alpha| <= max_depth and selected terminal actions.
    """
    if P.dim() != 4 or r.dim() != 3:
        raise ValueError("P must be (B,S,A,S) and r must be (B,S,A)")
    B, S, A, S2 = P.shape
    if S2 != S or r.shape != (B, S, A):
        raise ValueError(f"shape mismatch: P={tuple(P.shape)}, r={tuple(r.shape)}")

    words = all_action_words(A, max_depth, word_actions)
    terminal = list(range(A)) if terminal_actions is None else list(terminal_actions)
    device, dtype = P.device, P.dtype

    identity = torch.eye(S, device=device, dtype=dtype).unsqueeze(0).expand(B, S, S)
    cache: dict[tuple[int, ...], torch.Tensor] = {(): identity}

    def reached_distribution(word: tuple[int, ...]) -> torch.Tensor:
        if word in cache:
            return cache[word]
        prefix, action = word[:-1], word[-1]
        prefix_dist = reached_distribution(prefix)
        step = P[:, :, action, :]
        dist = torch.einsum("bst,btu->bsu", prefix_dist, step)
        cache[word] = dist
        return dist

    features = []
    for word in words:
        dist = reached_distribution(word)
        for action in terminal:
            r_action = r[:, :, action]
            features.append(torch.einsum("bst,bt->bs", dist, r_action))
    return torch.stack(features, dim=-1)


def aggregate_with_membership(
    P: torch.Tensor,
    r: torch.Tensor,
    gamma: torch.Tensor,
    W: torch.Tensor,
    eps: float = 1e-8,
):
    """Build the reduced MDP induced by row-stochastic memberships W."""
    B, S, A, _ = P.shape
    if W.dim() != 3 or W.shape[:2] != (B, S):
        raise ValueError(f"W must be (B,S,M), got {tuple(W.shape)}")

    mass = W.sum(dim=1).clamp_min(eps)
    r_agg = torch.einsum("bsm,bsa->bma", W, r) / mass.unsqueeze(-1)

    dest_membership = torch.einsum("bsat,btn->bsan", P, W)
    transition_mass = torch.einsum("bsm,bsan->bman", W, dest_membership)
    P_agg = transition_mass / mass[:, :, None, None]
    P_agg = P_agg.clamp_min(0.0)
    P_agg = P_agg / P_agg.sum(dim=-1, keepdim=True).clamp_min(eps)

    gamma_agg = torch.einsum("bs,bsm->bm", gamma, W)
    gamma_agg = gamma_agg.clamp_min(eps)
    gamma_agg = gamma_agg / gamma_agg.sum(dim=1, keepdim=True).clamp_min(eps)
    return P_agg, r_agg, gamma_agg, mass


def disaggregate_y(y_agg: torch.Tensor, W: torch.Tensor, n_actions: int) -> torch.Tensor:
    B, S, M = W.shape
    A = int(n_actions)
    y_agg_3d = y_agg.view(B, M, A)
    return torch.einsum("bsm,bma->bsa", W, y_agg_3d).reshape(B, S * A)


@dataclass(frozen=True)
class MarkovSignatureConfig:
    n_states: int
    n_actions: int
    n_clusters: int
    signature_depth: int = 1
    n_word_actions: int | None = 7
    n_terminal_actions: int | None = 7
    encoder_hidden: int = 32
    embed_dim: int = 16
    use_raw_state_index: bool = True
    normalize_signature: bool = True
    seed: int | None = None


def make_markov_signature_config_from_tensors(
    P: torch.Tensor,
    *,
    n_clusters: int,
    signature_depth: int = 1,
    n_word_actions: int | None = 7,
    n_terminal_actions: int | None = 7,
    encoder_hidden: int = 32,
    embed_dim: int = 16,
    use_raw_state_index: bool = True,
    normalize_signature: bool = True,
    seed: int | None = None,
) -> MarkovSignatureConfig:
    """Build a signature config directly from an MDP transition tensor."""
    if P.dim() != 4:
        raise ValueError(f"P must be (B,S,A,S), got {tuple(P.shape)}")
    _, n_states, n_actions, n_states_2 = P.shape
    if n_states != n_states_2:
        raise ValueError(f"P must be square in state dimensions, got {tuple(P.shape)}")
    return MarkovSignatureConfig(
        n_states=int(n_states),
        n_actions=int(n_actions),
        n_clusters=int(n_clusters),
        signature_depth=int(signature_depth),
        n_word_actions=n_word_actions,
        n_terminal_actions=n_terminal_actions,
        encoder_hidden=int(encoder_hidden),
        embed_dim=int(embed_dim),
        use_raw_state_index=bool(use_raw_state_index),
        normalize_signature=bool(normalize_signature),
        seed=seed,
    )


class MarkovSignatureSoftAgg(nn.Module):
    """MDP-aware soft aggregation via finite-horizon Markov signatures."""

    def __init__(self, config: MarkovSignatureConfig):
        super().__init__()
        self.config = config
        self.n_states = int(config.n_states)
        self.n_actions = int(config.n_actions)
        self.n_clusters = int(config.n_clusters)
        self.word_actions = choose_even_actions(config.n_actions, config.n_word_actions)
        self.terminal_actions = choose_even_actions(config.n_actions, config.n_terminal_actions)

        n_words = len(all_action_words(config.n_actions, config.signature_depth, self.word_actions))
        sig_dim = n_words * len(self.terminal_actions)
        extra_dim = 1 if config.use_raw_state_index else 0
        in_dim = sig_dim + extra_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, config.encoder_hidden),
            nn.ReLU(),
            nn.Linear(config.encoder_hidden, config.embed_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(config.n_clusters, config.embed_dim) * 0.1)

        if config.seed is not None:
            gen = torch.Generator().manual_seed(int(config.seed))
            with torch.no_grad():
                self.prototypes.copy_(
                    torch.randn(config.n_clusters, config.embed_dim, generator=gen) * 0.1
                )

    def compute_logits(self, P: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        B, S, A, _ = P.shape
        signature = compute_markov_signature(
            P,
            r,
            max_depth=self.config.signature_depth,
            word_actions=self.word_actions,
            terminal_actions=self.terminal_actions,
        )
        if self.config.normalize_signature:
            mean = signature.mean(dim=1, keepdim=True)
            std = signature.std(dim=1, keepdim=True).clamp_min(1e-6)
            signature = (signature - mean) / std
        if self.config.use_raw_state_index:
            idx = torch.linspace(0.0, 1.0, S, device=P.device, dtype=P.dtype)
            idx = idx.view(1, S, 1).expand(B, S, 1)
            encoder_input = torch.cat([signature, idx], dim=-1)
        else:
            encoder_input = signature
        h = self.encoder(encoder_input)
        diff = h.unsqueeze(2) - self.prototypes.view(1, 1, self.n_clusters, -1)
        return -diff.pow(2).sum(dim=-1)

    def membership(self, P: torch.Tensor, r: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        return F.softmax(self.compute_logits(P, r) / max(float(tau), 1e-8), dim=-1)

    def forward(self, P: torch.Tensor, r: torch.Tensor, gamma: torch.Tensor, tau: float = 0.5):
        start = time.time()
        logits = self.compute_logits(P, r)
        W = F.softmax(logits / max(float(tau), 1e-8), dim=-1)
        P_agg, r_agg, gamma_agg, mass = aggregate_with_membership(P, r, gamma, W)
        return P_agg, r_agg, gamma_agg, {
            "W": W,
            "mass": mass,
            "logits": logits,
            "runtime_sec": time.time() - start,
        }


class StochasticMarkovSignatureSoftAgg(MarkovSignatureSoftAgg):
    """Sample multiple Markov-signature memberships by perturbing logits."""

    def __init__(
        self,
        config: MarkovSignatureConfig,
        *,
        noise_std: float = 0.3,
        n_membership_samples: int = 3,
    ):
        super().__init__(config)
        self.noise_std = float(noise_std)
        self.n_membership_samples = max(1, int(n_membership_samples))

    def sample_aggregations(self, P: torch.Tensor, r: torch.Tensor, gamma: torch.Tensor, tau: float = 0.5):
        logits = self.compute_logits(P, r)
        samples = []
        start = time.time()
        for _ in range(self.n_membership_samples):
            noise = torch.randn_like(logits) * self.noise_std
            W = F.softmax((logits + noise) / max(float(tau), 1e-8), dim=-1)
            P_agg, r_agg, gamma_agg, mass = aggregate_with_membership(P, r, gamma, W)
            samples.append((P_agg, r_agg, gamma_agg, {"W": W, "mass": mass, "logits": logits}))
        return samples, {"runtime_sec": time.time() - start, "logits": logits}

    def forward(self, P: torch.Tensor, r: torch.Tensor, gamma: torch.Tensor, tau: float = 0.5):
        samples, info = self.sample_aggregations(P, r, gamma, tau=tau)
        P_agg, r_agg, gamma_agg, sample_info = samples[0]
        merged = dict(sample_info)
        merged.update(info)
        merged["samples"] = samples
        return P_agg, r_agg, gamma_agg, merged
