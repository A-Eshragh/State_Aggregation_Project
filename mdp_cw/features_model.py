"""
Neural models for the cliff-walking pipeline.

Pipeline (at training/test time):

    w  -- FeatureNet -->  h  -- (detach) -->  h_frozen  -- TransitionHead -->  slip_dist  -->  P_pred, r_pred

* FeatureNet is PRE-TRAINED to map weather w -> slip distribution
  (equivalently to P) via a standard supervised objective.  After that, its
  parameters are frozen and only the internal feature representation `h` is
  consumed downstream.
* TransitionHead is a single learnable matrix (+ softmax) mapping frozen
  features to slip-distribution logits.  This is the ONLY module that
  receives gradients during DFL / two-stage training on the decision head.

Output shapes
-------------
Both FeatureNet and TransitionHead work in a "compact slot" parameterisation:

    slots = 1 (ambient)                 for state-independent regimes (v1/v2)
    slots = 1 + |volatile_cells|        for v3 state-dependent regimes

Compact output: (B, slots, A, K). Slot 0 = ambient. Slots 1..V = overrides at
`volatile_cells[0..V-1]`.

TransitionHead also exposes `forward_full` which expands the compact output
to (B, S, A, K) by broadcasting ambient and patching the volatile cells.
`build_P_r_from_slip` in `cliff_mdp.py` accepts both (B, A, K) and (B, S, A, K)
so both modes flow through the rest of the pipeline unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mdp_cw.cliff_mdp import N_ACTIONS, N_SLIP, N_STATES, build_P_r_from_slip


# ---------------------------------------------------------------------------
# FeatureNet  —  pretrained encoder from weather features to transitions
# ---------------------------------------------------------------------------
class FeatureNet(nn.Module):
    """
    Two-stage architecture:

        encoder:  w (in_dim) -> h (feat_dim)
        head:     h -> slip logits (slots * A * K)

    Supervised target during pretraining: the true compact slip distribution
    for the weather sample.  After pretraining we freeze the whole network
    and only use `features(w)` downstream.

    Args:
        in_dim: raw weather feature dimension.
        feat_dim: size of the frozen feature vector `h`.
        hidden: hidden width of the encoder.
        n_slots: number of compact slip slots.  1 for v1/v2 (ambient only),
            1 + |volatile_cells| for v3.
    """

    def __init__(self, in_dim: int, feat_dim: int = 32, hidden: int = 64,
                 n_slots: int = 1):
        super().__init__()
        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {n_slots}")
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feat_dim),
            nn.Tanh(),
        )
        self.pretrain_head = nn.Linear(feat_dim, n_slots * N_ACTIONS * N_SLIP)
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.n_slots = n_slots

    def features(self, w: torch.Tensor) -> torch.Tensor:
        """Return (B, feat_dim) features."""
        return self.encoder(w)

    def forward(self, w: torch.Tensor):
        """
        Run full model (used during pretraining).

        Returns:
            slip_compact: (B, n_slots, A, K) softmaxed per (slot, action).
            h:            (B, feat_dim) features.
        """
        h = self.features(w)
        logits = self.pretrain_head(h).view(-1, self.n_slots, N_ACTIONS, N_SLIP)
        slip_compact = F.softmax(logits, dim=-1)
        return slip_compact, h

    def freeze(self) -> None:
        """Freeze all parameters and switch to eval mode (dropout/BN)."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()


# ---------------------------------------------------------------------------
# TransitionHead  —  learnable features -> slip distribution
# ---------------------------------------------------------------------------
class TransitionHead(nn.Module):
    """
    Single linear matrix M : feat_dim -> n_slots * A * K (slip logits).

    This is the 'another matrix' in the plan.  It is the only module that
    receives gradients when we train decision-focused / two-stage losses.

    Args:
        feat_dim: frozen feature dim.
        volatile_cells: optional list of state indices that get per-cell
            overrides.  If empty/None the head reduces to the legacy
            state-independent (A, K) output.
        n_states: number of MDP states (defaults to the cliff-walking
            constant; kept configurable only for testing).
    """

    def __init__(
        self,
        feat_dim: int,
        volatile_cells: Optional[list[int]] = None,
        n_states: int = N_STATES,
        ambient_only: bool = False,
    ):
        super().__init__()
        self.volatile_cells = list(volatile_cells) if volatile_cells else []
        self.n_slots = 1 + len(self.volatile_cells)
        self.n_states = n_states
        # Deliberate misspecification: the head predicts ONE ambient slip and
        # broadcasts it to every volatile slot, so the model class cannot
        # represent state-dependent hazards even though the environment has them.
        self.ambient_only = bool(ambient_only)
        self.linear = nn.Linear(feat_dim, self.n_slots * N_ACTIONS * N_SLIP, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    # -- Core output -----------------------------------------------------------
    def forward_compact(self, h: torch.Tensor) -> torch.Tensor:
        """Return (B, n_slots, A, K) softmaxed compact slip slots."""
        logits = self.linear(h).view(-1, self.n_slots, N_ACTIONS, N_SLIP)
        compact = F.softmax(logits, dim=-1)
        if self.ambient_only and self.n_slots > 1:
            compact = compact[:, :1].expand(-1, self.n_slots, -1, -1)
        return compact

    def forward_full(self, compact: torch.Tensor) -> torch.Tensor:
        """Expand compact (B, n_slots, A, K) to (B, S, A, K)."""
        B = compact.shape[0]
        ambient = compact[:, 0]                                  # (B, A, K)
        full = ambient.unsqueeze(1).expand(B, self.n_states, N_ACTIONS, N_SLIP).clone()
        for i, cell in enumerate(self.volatile_cells):
            full[:, cell] = compact[:, i + 1]
        return full

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Legacy scalar output.

        Returns:
            (B, A, K)     if state-independent (no volatile cells).
            (B, S, A, K)  if state-dependent.
        """
        compact = self.forward_compact(h)
        if not self.volatile_cells:
            return compact.squeeze(1)                             # (B, A, K)
        return self.forward_full(compact)                         # (B, S, A, K)


# ---------------------------------------------------------------------------
# Combined model  —  encodes the 'features -> transitions' pipeline
# ---------------------------------------------------------------------------
class CliffTransitionModel(nn.Module):
    """
    Wraps (frozen FeatureNet) + (learnable TransitionHead).

    forward(w) returns (P, r, slip_compact) where:
        P:            (B, S, A, S) predicted transitions
        r:            (B, S, A) predicted rewards
        slip_compact: (B, n_slots, A, K) used for CE / prediction loss

    Gradients flow into TransitionHead only.
    """

    def __init__(self, feature_net: FeatureNet, head: TransitionHead | None = None,
                 detach_features: bool = True):
        super().__init__()
        self.feature_net = feature_net
        self.head = head if head is not None else TransitionHead(feature_net.feat_dim)
        self.detach_features = detach_features

    @property
    def n_slots(self) -> int:
        return self.head.n_slots

    def compute_features(self, w: torch.Tensor) -> torch.Tensor:
        """Features from the frozen backbone (with gradient detached by default)."""
        if self.detach_features:
            with torch.no_grad():
                h = self.feature_net.features(w)
            return h.detach()
        return self.feature_net.features(w)

    def predict_slip(self, w: torch.Tensor):
        """Return (slip_for_mdp, slip_compact).

        slip_for_mdp: shape suitable for `build_P_r_from_slip`
            (B, A, K) if state-independent, (B, S, A, K) otherwise.
        slip_compact: (B, n_slots, A, K) — used for the prediction loss.
        """
        h = self.compute_features(w)
        compact = self.head.forward_compact(h)
        if not self.head.volatile_cells:
            return compact.squeeze(1), compact
        return self.head.forward_full(compact), compact

    def forward(self, w: torch.Tensor):
        slip_for_mdp, slip_compact = self.predict_slip(w)
        P, r = build_P_r_from_slip(slip_for_mdp)
        return P, r, slip_compact


__all__ = ["FeatureNet", "TransitionHead", "CliffTransitionModel"]
