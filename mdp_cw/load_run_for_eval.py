"""
Load a finished cliff run (weights + config) for analysis-only inference.

Supports ``twostage``, ``dfl``, and ``dfl_qpth``. ``dfl_ssa`` is not fully
restored (aggregator weights are not saved in the current training loop).
"""

from __future__ import annotations

from pathlib import Path

import torch
from qpth.qp import QPFunction, QPSolvers

from mdp_cw.cliff_run_utils import namespace_from_run_config, prepare_loaders_from_run_dir, read_run_config
from mdp_cw.features_model import CliffTransitionModel, FeatureNet, TransitionHead
from mdp_cw.train_cliff import _solve_with_algo, eval_y_on_true_mdp, _build_gamma
from mdp_lp.solvers.dense_lp_solver import DenseLPSolver


def load_stack_for_eval(
    run_dir: Path,
    device: torch.device | None = None,
):
    run_dir = Path(run_dir)
    if device is None:
        device = torch.device("cpu")

    rc = read_run_config(run_dir)
    args = namespace_from_run_config(rc)
    train_loader, test_loader, meta = prepare_loaders_from_run_dir(run_dir)

    w_sample, *_ = next(iter(train_loader))
    in_dim = int(w_sample.shape[-1])
    hidden = int(args.hidden_dim)
    feat_dim = int(args.feat_dim)
    vol_cells = list(meta.get("volatile_cells", []) or [])
    n_slots = 1 + len(vol_cells)

    feature_net = FeatureNet(
        in_dim=in_dim, feat_dim=feat_dim, hidden=hidden, n_slots=n_slots,
    ).to(device)
    head = TransitionHead(feat_dim=feat_dim, volatile_cells=vol_cells).to(device)
    model = CliffTransitionModel(feature_net, head, detach_features=True).to(device)

    if args.algo == "dfl_ssa":
        raise NotImplementedError(
            "dfl_ssa checkpoints do not persist SoftAggDense weights; "
            "use twostage / dfl / dfl_qpth runs for mechanism plots.",
        )

    agg = None
    sa_tau = float(getattr(args, "sa_tau", 0.5))
    fn = run_dir / "feature_net_state.pt"
    hd = run_dir / "head_state.pt"
    if not fn.exists() or not hd.exists():
        raise FileNotFoundError(f"Expected {fn} and {hd}")
    feature_net.load_state_dict(torch.load(fn, map_location=device, weights_only=True))
    head.load_state_dict(torch.load(hd, map_location=device, weights_only=True))
    feature_net.eval()
    head.eval()

    solver_obj = DenseLPSolver()

    def lp_solver(H, r, gamma):
        return solver_obj.solve(H, r, gamma)

    qp_solver = QPFunction(eps=1e-10, maxIter=20, solver=QPSolvers.PDIPM_BATCHED)
    return args, model, agg, test_loader, train_loader, meta, lp_solver, qp_solver, sa_tau


def greedy_policy_from_occupancy(y_star: torch.Tensor, n_states: int = 48, n_actions: int = 4):
    """Argmax action from occupancy weights (S, A)."""
    y = y_star.view(n_states, n_actions)
    d = y.sum(dim=-1, keepdim=True)
    pi = y / (d + 1e-12)
    return pi.argmax(dim=-1)


__all__ = ["greedy_policy_from_occupancy", "load_stack_for_eval"]
