"""
Weather -> transition dynamics for Cliff-Walking.

Data-generation direction: we start from latent weather regimes z,
choose a slip distribution P(z) per regime (ground truth), then
sample observable weather features w | z (noisy encoding).

At train/test time the model learns the FORWARD direction w -> P.
This file only provides the ground-truth generator (z -> slip-dist,
z -> w).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from mdp_cw.cliff_mdp import (
    N_ACTIONS,
    N_SLIP,
    N_STATES,
    build_P_r_from_slip,
    state_idx,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Slip kinds: 0=intended, 1=left(-90°), 2=right(+90°), 3=back(180°), 4=no-move.
#
# Cliff geometry (cliff = row 3, cols 1..10). Which slip kinds are dangerous:
#   RIGHT action + slip-RIGHT  -> DOWN  (cliff risk from row 2)
#   UP    action + slip-BACK   -> DOWN  (cliff risk from row 2)
#   RIGHT action + slip-LEFT   -> UP    (safe drift)
# So regimes pumping (right-slip on RIGHT) or (back-slip on UP) force the
# optimal path to abandon row 2 and hug row 0/1.

def _row(intended: float, left: float, right: float, back: float, no_move: float):
    total = intended + left + right + back + no_move
    return [intended / total, left / total, right / total, back / total, no_move / total]


# ---------------------------------------------------------------------------
# Regime library v1 (original, 4 regimes — produce ~same optimal policy)
# ---------------------------------------------------------------------------
CLEAR = [_row(0.94, 0.02, 0.02, 0.00, 0.02)] * N_ACTIONS

WINDY = [
    _row(0.70, 0.05, 0.20, 0.00, 0.05),
    _row(0.85, 0.03, 0.03, 0.03, 0.06),
    _row(0.70, 0.05, 0.20, 0.00, 0.05),
    _row(0.55, 0.15, 0.15, 0.05, 0.10),
]

RAINY = [_row(0.70, 0.10, 0.10, 0.02, 0.08)] * N_ACTIONS

ICY = [_row(0.45, 0.20, 0.20, 0.05, 0.10)] * N_ACTIONS

REGIMES_V1: list[list[list[float]]] = [CLEAR, WINDY, RAINY, ICY]
REGIME_NAMES_V1: list[str] = ["clear", "windy", "rainy", "icy"]


# ---------------------------------------------------------------------------
# Regime library v2 — 8 policy-diverse regimes grouped into two families.
#
# SAFE family (row 2 is viable and shortest -> optimal path hugs row 2):
#   0 CLEAR_A, 1 CLEAR_B, 2 SOUTHWIND_MILD, 3 NORTHWIND
# DANGER family (row 2 is catastrophic -> optimal path is row 0/1):
#   4 SOUTHWIND_STRONG, 5 DOWNDRAFT, 6 BACKDRAFT, 7 ICE_STORM
#
# Pairs (0,1) and (2,3) can be made feature-ambiguous via ambiguous_pairs,
# so TS *must* misclassify some SAFE as DANGER (or vice versa), which is
# exactly when DFL's decision-awareness can pay off.
# ---------------------------------------------------------------------------
_SAFE_GEN = _row(0.88, 0.05, 0.05, 0.00, 0.02)

REGIMES_V2: list[list[list[float]]] = [
    # 0 CLEAR_A — near-deterministic. Optimal: row 2.
    [_row(0.95, 0.02, 0.02, 0.00, 0.01)] * N_ACTIONS,
    # 1 CLEAR_B — near-deterministic twin (intended feature-ambiguous with CLEAR_A).
    [_row(0.93, 0.03, 0.03, 0.00, 0.01)] * N_ACTIONS,
    # 2 SOUTHWIND_MILD — RIGHT action slips south (=DOWN) 15% -> mild cliff risk on row 2, row 2 still wins.
    [_SAFE_GEN, _row(0.70, 0.10, 0.15, 0.00, 0.05), _SAFE_GEN, _SAFE_GEN],
    # 3 NORTHWIND — RIGHT action slips north (=UP) 25% -> drifts safely away from cliff, row 2 OK.
    [_SAFE_GEN, _row(0.65, 0.25, 0.05, 0.00, 0.05), _SAFE_GEN, _SAFE_GEN],
    # 4 SOUTHWIND_STRONG — RIGHT action slips south 30% -> row 2 suicidal, row 0 optimal.
    [_SAFE_GEN, _row(0.55, 0.05, 0.30, 0.00, 0.10), _SAFE_GEN, _SAFE_GEN],
    # 5 DOWNDRAFT — UP action slips back (=DOWN) 25% -> climbing from row 3 is unreliable, row 2 risky.
    [_row(0.60, 0.05, 0.05, 0.25, 0.05), _SAFE_GEN, _SAFE_GEN, _SAFE_GEN],
    # 6 BACKDRAFT — all actions back-slip 20% (general un-commitment). Row 0 safest.
    [_row(0.65, 0.05, 0.05, 0.20, 0.05)] * N_ACTIONS,
    # 7 ICE_STORM — heavy stochasticity everywhere. Row 0 strictly safest.
    [_row(0.40, 0.15, 0.15, 0.15, 0.15)] * N_ACTIONS,
]
REGIME_NAMES_V2: list[str] = [
    "clear_A", "clear_B", "southwind_mild", "northwind",
    "southwind_strong", "downdraft", "backdraft", "ice_storm",
]

# Pairs that should be made feature-ambiguous (one SAFE, one DANGER).
# Tuning knob: passing these as `ambiguous_pairs` collapses their cluster
# centers so TS cannot reliably tell them apart -> DFL's edge shows up.
DEFAULT_V2_AMBIGUOUS_PAIRS: list[tuple[int, int]] = [
    (0, 5),   # CLEAR_A    <-> DOWNDRAFT        (same-looking features, very different cliff risk)
    (2, 4),   # SOUTHWIND_MILD <-> SOUTHWIND_STRONG (mild vs suicidal)
]


# ---------------------------------------------------------------------------
# Regime library v3 — state-DEPENDENT slip via sparse "volatile cells".
#
# Motivation:
#   In v1/v2 the slip distribution is the same at every state, so the
#   optimal policy is determined by the *global* weather pattern. This
#   makes the decision boundary smooth: small mispredictions of the
#   ambient slip barely change the policy, and TS's averaged prediction
#   is already near-optimal. DFL has no leverage.
#
#   v3 introduces SPARSE volatile cells along row 2 (the cliff-adjacent
#   row). Each regime has a benign "ambient" slip that applies
#   everywhere, plus a per-cell override at specific volatile cells.
#   Because the icy cell is *localised*, the optimal policy depends on
#   *which* cell is icy -- not just "is weather bad on average?". TS
#   that averages over regimes smears the hazard across all volatile
#   cells; DFL can learn a sharper, regime-specific routing.
#
# Regime representation:
#   dict with keys
#     'ambient':   list[list[float]] of shape (A, K)
#     'overrides': dict[int -> list[list[float]]] mapping
#                  state index -> (A, K) override for that cell.
#
# Default volatile cells: three on row 2 (close to the cliff).
# ---------------------------------------------------------------------------
VOLATILE_CELLS_V3: list[int] = [state_idx(2, 3), state_idx(2, 6), state_idx(2, 9)]

# Cliff-geometry refresher (from row 2, about to move along the cliff-adjacent
# row):
#   RIGHT + right-slip  -> rotates (0,+1) by +90° = (+1, 0) = DOWN -> cliff
#   UP    + back-slip   -> rotates (-1,0) by 180° = (+1, 0) = DOWN -> cliff
# These are the ONLY slip outcomes that teleport us back to start.
# So "safe on row 2" means: zero right-slip on RIGHT and zero back-slip on UP.

# Ambient: near-deterministic on row-2-relevant (action, slip) combos.
_AMBIENT_BENIGN = [
    # UP:    no back-slip (the cliff-seeking one). Residual slip is safe.
    _row(0.95, 0.02, 0.02, 0.00, 0.01),
    # RIGHT: no right-slip (the cliff-seeking one). Small back-slip = LEFT = safe.
    _row(0.95, 0.02, 0.00, 0.02, 0.01),
    # DOWN:  rarely used from row 2 (you'd just go into the cliff). Kept mild.
    _row(0.90, 0.04, 0.04, 0.00, 0.02),
    # LEFT:  all slips are safe on row 2.
    _row(0.95, 0.02, 0.02, 0.00, 0.01),
]

# Icy override at a volatile cell: massive cliff-seeking slip for BOTH
# RIGHT and UP actions, so you can neither push through nor commit
# upward from an icy cell without very high cliff risk. The only way
# to handle an icy cell is to detour BEFORE entering it: from the cell
# to its left, go UP to row 1, traverse, then DOWN on the other side.
_ICY_OVERRIDE = [
    # UP:    55% back-slip -> DOWN -> cliff.
    _row(0.25, 0.10, 0.10, 0.55, 0.00),
    # RIGHT: 55% right-slip -> DOWN -> cliff.
    _row(0.25, 0.10, 0.55, 0.10, 0.00),
    # DOWN:  direct into cliff.
    _row(0.60, 0.10, 0.10, 0.10, 0.10),
    # LEFT:  safer (slip away from cliff).
    _row(0.70, 0.10, 0.10, 0.05, 0.05),
]


def _regime_v3(ambient, overrides_by_col_cell):
    """Helper: ambient + {(row, col): override_matrix}."""
    return {
        "ambient": ambient,
        "overrides": {state_idx(r, c): override
                      for (r, c), override in overrides_by_col_cell.items()},
    }


REGIMES_V3: list[dict] = [
    # 0 CLEAR — no icy cells, ambient mild. Optimal: row 2 direct.
    _regime_v3(_AMBIENT_BENIGN, {}),
    # 1 ICE_W — only (2, 3) is icy.
    _regime_v3(_AMBIENT_BENIGN, {(2, 3): _ICY_OVERRIDE}),
    # 2 ICE_M — only (2, 6) is icy.
    _regime_v3(_AMBIENT_BENIGN, {(2, 6): _ICY_OVERRIDE}),
    # 3 ICE_E — only (2, 9) is icy.
    _regime_v3(_AMBIENT_BENIGN, {(2, 9): _ICY_OVERRIDE}),
]
REGIME_NAMES_V3: list[str] = ["clear", "ice_west", "ice_mid", "ice_east"]


# ---------------------------------------------------------------------------
# Regime library v4 — same 4 semantic regimes as v3, but **more volatile sites**
# and **multi-cell** ICE patterns per regime (harder state-dependent routing).
# Union of volatile cells (slot order) is fixed; regimes only override subsets;
# missing slots use ambient in the compact head (same contract as v3).
# ---------------------------------------------------------------------------
_VOL_V4_CELLS = [
    (1, 6),  # above mid corridor — second hazard in ice_mid
    (2, 3),  # west patch
    (2, 4),  # adjacent west — dual-icy strip for ice_west
    (2, 6),  # mid
    (2, 8),  # east pair
    (2, 9),
]
VOLATILE_CELLS_V4: list[int] = sorted({state_idx(r, c) for (r, c) in _VOL_V4_CELLS})

REGIMES_V4: list[dict] = [
    _regime_v3(_AMBIENT_BENIGN, {}),
    _regime_v3(
        _AMBIENT_BENIGN,
        {(2, 3): _ICY_OVERRIDE, (2, 4): _ICY_OVERRIDE},
    ),
    _regime_v3(
        _AMBIENT_BENIGN,
        {(2, 6): _ICY_OVERRIDE, (1, 6): _ICY_OVERRIDE},
    ),
    _regime_v3(
        _AMBIENT_BENIGN,
        {(2, 8): _ICY_OVERRIDE, (2, 9): _ICY_OVERRIDE},
    ),
]
REGIME_NAMES_V4: list[str] = list(REGIME_NAMES_V3)


# ---------------------------------------------------------------------------
# Regime library v5 — deliberately decision-sharp ambiguous clear-vs-hazard.
#
# v4 makes the hazard location differ by regime. That is realistic, but under
# heavy feature ambiguity it leaves DFL with no single safe correction: west,
# mid, and east ice require different local detours. v5 instead makes all
# non-clear regimes share a common cliff-adjacent hazard strip. The useful
# decision is binary and global: use the short row-2 path in clear weather, or
# leave row 2 early and traverse above it when the strip is risky.
# ---------------------------------------------------------------------------
VOLATILE_CELLS_V5: list[int] = [state_idx(2, c) for c in range(1, 11)]

_STRIP_OVERRIDE = [
    # UP: back-slip falls toward the cliff from row 2.
    _row(0.15, 0.05, 0.05, 0.75, 0.00),
    # RIGHT: right-slip falls toward the cliff from row 2.
    _row(0.15, 0.05, 0.75, 0.05, 0.00),
    # DOWN: directly cliff-seeking from row 2.
    _row(0.45, 0.10, 0.15, 0.15, 0.15),
    # LEFT: mostly safe retreat.
    _row(0.80, 0.05, 0.05, 0.05, 0.05),
]

_STRIP_OVERRIDE_MILD = [
    _row(0.25, 0.08, 0.08, 0.59, 0.00),
    _row(0.25, 0.08, 0.59, 0.08, 0.00),
    _row(0.50, 0.10, 0.15, 0.15, 0.10),
    _row(0.78, 0.06, 0.06, 0.05, 0.05),
]

_STRIP_OVERRIDE_SEVERE = [
    _row(0.10, 0.05, 0.05, 0.80, 0.00),
    _row(0.10, 0.05, 0.80, 0.05, 0.00),
    _row(0.40, 0.10, 0.20, 0.20, 0.10),
    _row(0.82, 0.04, 0.04, 0.05, 0.05),
]


def _strip_regime(override):
    return _regime_v3(_AMBIENT_BENIGN, {(2, c): override for c in range(1, 11)})


REGIMES_V5: list[dict] = [
    _regime_v3(_AMBIENT_BENIGN, {}),
    _strip_regime(_STRIP_OVERRIDE_MILD),
    _strip_regime(_STRIP_OVERRIDE),
    _strip_regime(_STRIP_OVERRIDE_SEVERE),
]
REGIME_NAMES_V5: list[str] = ["clear", "strip_mild", "strip", "strip_severe"]


# v6 — one shared late gate trap. This keeps the solver small (2 slots) while
# making the risky direct route deceptively attractive under clear-heavy data.
VOLATILE_CELLS_V6: list[int] = [state_idx(2, 9)]

REGIMES_V6: list[dict] = [
    _regime_v3(_AMBIENT_BENIGN, {}),
    _regime_v3(_AMBIENT_BENIGN, {(2, 9): _STRIP_OVERRIDE_MILD}),
    _regime_v3(_AMBIENT_BENIGN, {(2, 9): _STRIP_OVERRIDE}),
    _regime_v3(_AMBIENT_BENIGN, {(2, 9): _STRIP_OVERRIDE_SEVERE}),
]
REGIME_NAMES_V6: list[str] = ["clear", "gate_mild", "gate", "gate_severe"]


# v7 — v6 geometry with a NEAR-ABSORBING severe gate. Designed for the
# in-distribution mean-planning trap: under a clear-heavy prior the
# posterior-mean slip at the gate looks retryable (independent per attempt),
# but the regime persists within an episode, so on gate_deadly samples the
# deployed gate policy enters a discounted reset loop it almost never exits.
# Planning on the posterior-mean kernel is therefore systematically optimistic
# at the gate, while an always-detour policy is Bayes-optimal — a gap DFL can
# exploit with NO train/test distribution shift (use beta ~0.99 so the reset
# loop is not discounted away; the analytic gap peaks near a 5:1 clear prior).
VOLATILE_CELLS_V7: list[int] = list(VOLATILE_CELLS_V6)

_GATE_DEADLY = [
    _row(0.03, 0.01, 0.01, 0.95, 0.00),  # UP: back-slip into the cliff row
    _row(0.03, 0.01, 0.95, 0.01, 0.00),  # RIGHT: right-slip into the cliff row
    _row(0.90, 0.02, 0.04, 0.02, 0.02),  # DOWN: cliff-seeking anyway
    _row(0.80, 0.05, 0.05, 0.05, 0.05),  # LEFT: mostly safe retreat
]

REGIMES_V7: list[dict] = [
    _regime_v3(_AMBIENT_BENIGN, {}),
    _regime_v3(_AMBIENT_BENIGN, {(2, 9): _STRIP_OVERRIDE_MILD}),
    _regime_v3(_AMBIENT_BENIGN, {(2, 9): _STRIP_OVERRIDE}),
    _regime_v3(_AMBIENT_BENIGN, {(2, 9): _GATE_DEADLY}),
]
REGIME_NAMES_V7: list[str] = ["clear", "gate_mild", "gate", "gate_deadly"]


# ---------------------------------------------------------------------------
# Active regime set (mutable; switch via set_regime_set before data gen).
# ---------------------------------------------------------------------------
REGIMES: list = REGIMES_V1
REGIME_NAMES: list[str] = REGIME_NAMES_V1
VOLATILE_CELLS: list[int] = []
_ACTIVE_MODE: str = "v1"


def set_regime_set(name: str) -> None:
    """Switch the active regime library. Call once before data generation."""
    global REGIMES, REGIME_NAMES, VOLATILE_CELLS, _ACTIVE_MODE
    name = name.lower()
    if name in ("v1", "original"):
        REGIMES, REGIME_NAMES = REGIMES_V1, REGIME_NAMES_V1
        VOLATILE_CELLS = []
        _ACTIVE_MODE = "v1"
    elif name == "v2":
        REGIMES, REGIME_NAMES = REGIMES_V2, REGIME_NAMES_V2
        VOLATILE_CELLS = []
        _ACTIVE_MODE = "v2"
    elif name == "v3":
        REGIMES, REGIME_NAMES = REGIMES_V3, REGIME_NAMES_V3
        VOLATILE_CELLS = list(VOLATILE_CELLS_V3)
        _ACTIVE_MODE = "v3"
    elif name == "v4":
        REGIMES, REGIME_NAMES = REGIMES_V4, REGIME_NAMES_V4
        VOLATILE_CELLS = list(VOLATILE_CELLS_V4)
        _ACTIVE_MODE = "v4"
    elif name == "v5":
        REGIMES, REGIME_NAMES = REGIMES_V5, REGIME_NAMES_V5
        VOLATILE_CELLS = list(VOLATILE_CELLS_V5)
        _ACTIVE_MODE = "v5"
    elif name == "v6":
        REGIMES, REGIME_NAMES = REGIMES_V6, REGIME_NAMES_V6
        VOLATILE_CELLS = list(VOLATILE_CELLS_V6)
        _ACTIVE_MODE = "v6"
    elif name == "v7":
        REGIMES, REGIME_NAMES = REGIMES_V7, REGIME_NAMES_V7
        VOLATILE_CELLS = list(VOLATILE_CELLS_V7)
        _ACTIVE_MODE = "v7"
    else:
        raise ValueError(f"unknown regime_set '{name}' (use 'v1'..'v7')")


def is_state_dependent() -> bool:
    """True if the active regime set encodes per-cell slip overrides."""
    return _ACTIVE_MODE in ("v3", "v4", "v5", "v6", "v7")


def n_regimes() -> int:
    return len(REGIMES)


def volatile_cells() -> list[int]:
    """Return the list of state indices that carry a per-cell override."""
    return list(VOLATILE_CELLS)


def n_slots() -> int:
    """Number of distinct slip 'slots' per regime: 1 (ambient) + |volatile|.

    For v1/v2 this is always 1. For v3 it matches the number of volatile
    cells plus the ambient slot.
    """
    return 1 + len(VOLATILE_CELLS)


# --- Per-regime lookups: legacy (state-independent) ---------------------------
def slip_dist_for_regime(regime_id: int) -> torch.Tensor:
    """Return the (A, K) ambient slip distribution for a regime.

    For v1/v2 this is the (only) slip distribution. For v3 it is the
    ambient slot (ignoring per-cell overrides).
    """
    reg = REGIMES[regime_id]
    ambient = reg["ambient"] if isinstance(reg, dict) else reg
    return torch.tensor(ambient, dtype=torch.float32)


def slip_dist_batch(regime_ids: torch.Tensor) -> torch.Tensor:
    """(B,) long tensor of regime ids -> (B, A, K) ambient slip distributions."""
    table = torch.stack([slip_dist_for_regime(i) for i in range(n_regimes())], dim=0)
    return table[regime_ids]


# --- Per-regime lookups: full (S, A, K) and compact (n_slots, A, K) ----------
def slip_dist_full_for_regime(regime_id: int) -> torch.Tensor:
    """Return the (S, A, K) slip distribution for a regime.

    For v1/v2 this is the ambient slip broadcast across all states. For
    v3 it is the ambient slip with per-cell overrides applied at every
    volatile cell.
    """
    reg = REGIMES[regime_id]
    if isinstance(reg, dict):
        ambient = torch.tensor(reg["ambient"], dtype=torch.float32)        # (A, K)
        full = ambient.unsqueeze(0).expand(N_STATES, N_ACTIONS, N_SLIP).clone()
        for cell_idx, override in reg["overrides"].items():
            full[cell_idx] = torch.tensor(override, dtype=torch.float32)
    else:
        ambient = torch.tensor(reg, dtype=torch.float32)
        full = ambient.unsqueeze(0).expand(N_STATES, N_ACTIONS, N_SLIP).clone()
    return full


def slip_dist_full_batch(regime_ids: torch.Tensor) -> torch.Tensor:
    """(B,) -> (B, S, A, K)."""
    table = torch.stack([slip_dist_full_for_regime(i) for i in range(n_regimes())], dim=0)
    return table[regime_ids]


def slip_dist_compact_for_regime(regime_id: int) -> torch.Tensor:
    """Return the compact (n_slots, A, K) slip representation.

    Slot 0 is ambient. For v3 the next |V| slots are the per-cell
    slip distributions at VOLATILE_CELLS (in order). For v1/v2
    n_slots=1 so the compact tensor is just ambient unsqueezed.
    """
    reg = REGIMES[regime_id]
    if isinstance(reg, dict):
        ambient = torch.tensor(reg["ambient"], dtype=torch.float32)         # (A, K)
        slots = [ambient]
        for cell in VOLATILE_CELLS:
            if cell in reg["overrides"]:
                slots.append(torch.tensor(reg["overrides"][cell], dtype=torch.float32))
            else:
                slots.append(ambient)
        return torch.stack(slots, dim=0)                                   # (1+V, A, K)
    else:
        ambient = torch.tensor(reg, dtype=torch.float32)
        return ambient.unsqueeze(0)                                         # (1, A, K)


def slip_dist_compact_batch(regime_ids: torch.Tensor) -> torch.Tensor:
    """(B,) -> (B, n_slots, A, K)."""
    table = torch.stack([slip_dist_compact_for_regime(i) for i in range(n_regimes())], dim=0)
    return table[regime_ids]


# ---------------------------------------------------------------------------
# Feature model:  z -> w
# ---------------------------------------------------------------------------
@dataclass
class FeatureGenConfig:
    """Configuration for sampling observable features from the latent regime."""
    feat_dim: int = 8
    noise_std: float = 0.6         # per-feature gaussian noise
    center_scale: float = 2.0      # radius of regime cluster centers; smaller = more overlap
    ambiguous_pairs: Optional[list[tuple[int, int]]] = None
    ambiguous_gap: float = 0.15    # cluster-center offset for ambiguous pairs
    seed: Optional[int] = None


def _make_regime_centers(
    feat_dim: int,
    n_reg: int,
    center_scale: float = 2.0,
    ambiguous_pairs: Optional[list[tuple[int, int]]] = None,
    ambiguous_gap: float = 0.15,
) -> torch.Tensor:
    """
    Deterministic cluster centers in feature space, one per regime.
    Each center is a random unit vector scaled to `center_scale`
    (smaller -> more overlap between clusters).

    If `ambiguous_pairs` is given, for each (i, j) the center of j is
    overwritten to sit very close to the center of i (offset by
    `ambiguous_gap` in a random direction). This makes the two regimes
    hard to distinguish from features alone.
    """
    g = torch.Generator().manual_seed(1337)
    centers = torch.randn(n_reg, feat_dim, generator=g)
    centers = centers / centers.norm(dim=1, keepdim=True).clamp_min(1e-8)
    centers = centers * float(center_scale)

    if ambiguous_pairs:
        for i, j in ambiguous_pairs:
            direction = torch.randn(feat_dim, generator=g)
            direction = direction / direction.norm().clamp_min(1e-8)
            centers[j] = centers[i] + float(ambiguous_gap) * direction
    return centers


def sample_features(regime_ids: torch.Tensor, cfg: FeatureGenConfig) -> torch.Tensor:
    """Sample observable weather features given regime ids."""
    centers = _make_regime_centers(
        cfg.feat_dim, n_regimes(),
        center_scale=cfg.center_scale,
        ambiguous_pairs=cfg.ambiguous_pairs,
        ambiguous_gap=cfg.ambiguous_gap,
    )
    mu = centers[regime_ids]
    if cfg.seed is not None:
        g = torch.Generator().manual_seed(cfg.seed)
        noise = torch.randn(mu.shape, generator=g) * cfg.noise_std
    else:
        noise = torch.randn_like(mu) * cfg.noise_std
    return mu + noise


# ---------------------------------------------------------------------------
# End-to-end ground truth: regime -> (slip_dist, P, r)
# ---------------------------------------------------------------------------
def ground_truth_mdp_tensors(regime_ids: torch.Tensor):
    """
    Given (B,) regime ids, return:
        slip_compact: (B, n_slots, A, K)  -- ambient slot + optional per-volatile
        P:            (B, S, A, S)        -- transition kernel (built from full slip)
        r:            (B, S, A)           -- expected per-state reward

    NOTE: the return slot name changed from `slip_dist` (B, A, K) to
    `slip_compact` (B, n_slots, A, K). For v1/v2 n_slots=1 so
    `slip_compact.squeeze(1)` recovers the legacy (B, A, K) tensor.
    """
    slip_compact = slip_dist_compact_batch(regime_ids)
    slip_full = slip_dist_full_batch(regime_ids)
    P, r = build_P_r_from_slip(slip_full)
    return slip_compact, P, r


def sample_regimes(n_samples: int, seed: Optional[int] = None, weights=None,
                   stratified: bool = False) -> torch.Tensor:
    """Draw regime ids. Optional `weights` is a (Z,) non-negative vector.

    With ``stratified=True`` the regime counts match the weights exactly (largest
    remainder rounding) and are then shuffled, removing the binomial draw noise.
    The population distribution is unchanged; only the sampling variance is.
    """
    Z = n_regimes()
    if weights is None:
        probs = torch.full((Z,), 1.0 / Z)
    else:
        w = torch.as_tensor(weights, dtype=torch.float32)
        if w.numel() != Z:
            raise ValueError(f"weights must have length {Z}, got {w.numel()}")
        probs = w / w.sum()

    if stratified:
        exact = probs * float(n_samples)
        counts = torch.floor(exact).to(torch.long)
        rem = int(n_samples - int(counts.sum().item()))
        if rem > 0:
            order = torch.argsort(exact - counts.float(), descending=True)
            for i in range(rem):
                counts[order[i % Z]] += 1
        ids = torch.repeat_interleave(torch.arange(Z, dtype=torch.long), counts)
        g = torch.Generator().manual_seed(seed if seed is not None else 0)
        return ids[torch.randperm(int(n_samples), generator=g)]

    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        return torch.multinomial(probs, n_samples, replacement=True, generator=g)
    return torch.multinomial(probs, n_samples, replacement=True)


__all__ = [
    "REGIMES", "REGIME_NAMES", "VOLATILE_CELLS",
    "REGIMES_V1", "REGIME_NAMES_V1",
    "REGIMES_V2", "REGIME_NAMES_V2", "DEFAULT_V2_AMBIGUOUS_PAIRS",
    "REGIMES_V3", "REGIME_NAMES_V3", "VOLATILE_CELLS_V3",
    "REGIMES_V4", "REGIME_NAMES_V4", "VOLATILE_CELLS_V4",
    "REGIMES_V5", "REGIME_NAMES_V5", "VOLATILE_CELLS_V5",
    "REGIMES_V6", "REGIME_NAMES_V6", "VOLATILE_CELLS_V6",
    "set_regime_set", "n_regimes", "is_state_dependent",
    "volatile_cells", "n_slots",
    "slip_dist_for_regime", "slip_dist_batch",
    "slip_dist_full_for_regime", "slip_dist_full_batch",
    "slip_dist_compact_for_regime", "slip_dist_compact_batch",
    "FeatureGenConfig", "sample_features",
    "sample_regimes", "ground_truth_mdp_tensors",
]
