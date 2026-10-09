# lp/dense_lp_solver.py

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB


# One Gurobi environment PER THREAD, created lazily. Reusing an Env avoids
# per-solve license/init overhead, which dominates for the small LPs solved
# inside DFL / DFL-sketch / DFL-berthet -- but a single Env shared across the
# pool serializes optimize(), so the batch solved effectively single-threaded
# no matter how many workers were configured. Measured on the M=12 sketch LP
# (8 rows x 48 cols, 256 per batch): shared Env gives 1.01x at 16 workers,
# per-thread Env gives 5.85x. The pool is a process-wide singleton, so at most
# MDPLP_LP_WORKERS + 1 environments are ever created.
_gurobi_tls = threading.local()


def _get_gurobi_env():
    env = getattr(_gurobi_tls, "env", None)
    if env is None:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        _gurobi_tls.env = env
    return env


# Shared thread pool for parallel batch solves. Each thread builds its own
# Model from its own Env, which is what makes the solves actually concurrent.
_executor = None
_executor_lock = threading.Lock()


def _get_executor():
    global _executor
    if _executor is None:
        with _executor_lock:
            if _executor is None:
                default_workers = min(os.cpu_count() or 4, 16)
                n_workers = int(os.environ.get("MDPLP_LP_WORKERS", default_workers))
                n_workers = max(1, n_workers)
                _executor = ThreadPoolExecutor(max_workers=n_workers)
    return _executor


def _solve_one_lp(H_i, r_i, gamma_i):
    """Solve one LP given numpy inputs. Designed to run inside a worker thread."""
    S, SA = H_i.shape
    env = _get_gurobi_env()
    model = gp.Model(env=env)
    # Single-threaded Gurobi per solve so the outer ThreadPool gets clean
    # parallelism without thread oversubscription. Dual simplex (Method=1) is
    # typically fastest for small LPs with warm-startable bases like ours.
    model.Params.Threads = 1
    model.Params.Method = 1

    try:
        y = model.addMVar(shape=SA, lb=0.0, vtype=GRB.CONTINUOUS)
        model.setObjective(r_i @ y, GRB.MAXIMIZE)
        model.addMConstr(H_i, y, "=", gamma_i)
        model.optimize()

        if model.status != GRB.OPTIMAL:
            raise RuntimeError("Gurobi failed to solve LP.")

        y_star = y.X
        vb = np.asarray(y.VBasis)
        rc = np.asarray(y.RC)

        # Primary: structural variables Gurobi marks as basic (VBasis == 0).
        basic_indices = np.flatnonzero(vb == 0).tolist()

        if len(basic_indices) >= S:
            chosen = basic_indices[:S]
        else:
            # Degenerate optimum: an artificial/slack stayed basic at zero, so
            # fewer than S structural variables have VBasis == 0. Pad with
            # non-basic structurals ordered by smallest |reduced cost| — most
            # likely to keep H_B well-conditioned for the backward pass.
            already = set(basic_indices)
            candidates = [j for j in range(SA) if j not in already]
            candidates.sort(key=lambda j: (abs(float(rc[j])), j))
            chosen = basic_indices + candidates[:S - len(basic_indices)]

        if len(chosen) != S:
            raise RuntimeError(
                f"Basis construction produced {len(chosen)} indices, expected {S}. "
                f"SA={SA}, |basic|={len(basic_indices)}."
            )

        basis_idx = torch.tensor(chosen, dtype=torch.long)

        constrs = model.getConstrs()
        pi_vals = []
        for c in constrs:
            p = c.Pi
            if hasattr(p, "__len__") and not isinstance(p, str):
                pi_vals.extend(np.atleast_1d(p).flatten().tolist())
            else:
                pi_vals.append(float(p))
        dual = torch.tensor(pi_vals, dtype=torch.float32)

        return torch.tensor(y_star, dtype=torch.float32), basis_idx, dual
    finally:
        model.dispose()


class DenseLPSolver:
    """
    Dense LP solver using Gurobi.
    Solves the discounted MDP LP for each batch element:

        maximize     r^T y
        subject to   H y = gamma
                     y >= 0

    Inputs:
        H:      (batch, S, SA)
        r:      (batch, SA)
        gamma:  (batch, S)

    Outputs:
        y_star: (batch, SA)
        basis:  (batch, S)   # basis column indices
        dual:   (batch, S)   # dual values (lambda) for H y = gamma
    """

    def __init__(self):
        pass

    def solve_single(self, H_i, r_i, gamma_i):
        """Solve ONE LP using Gurobi (kept for backward compatibility)."""
        return _solve_one_lp(H_i, r_i, gamma_i)

    def solve(self, H, r, gamma):
        """
        Solve LP for the whole batch. Per-batch solves run on a shared
        ThreadPool so they execute concurrently (Gurobi releases the GIL).

        Returns:
            y_star: (batch, SA)
            basis:  (batch, S)
            dual:   (batch, S)
        """
        batch, S, SA = H.shape

        H_np = H.detach().cpu().numpy()
        r_np = r.detach().cpu().numpy()
        gamma_np = gamma.detach().cpu().numpy()

        if batch == 1:
            y_b, B_b, dual_b = _solve_one_lp(H_np[0], r_np[0], gamma_np[0])
            return y_b.unsqueeze(0), B_b.unsqueeze(0), dual_b.unsqueeze(0)

        executor = _get_executor()
        futures = [
            executor.submit(_solve_one_lp, H_np[b], r_np[b], gamma_np[b])
            for b in range(batch)
        ]
        results = [f.result() for f in futures]

        ys = [res[0] for res in results]
        bases = [res[1] for res in results]
        duals = [res[2] for res in results]

        bad = [(b, tuple(bases[b].shape)) for b in range(batch) if bases[b].shape != (S,)]
        if bad:
            raise RuntimeError(
                f"DenseLPSolver.solve: heterogeneous basis shapes detected. "
                f"Expected ({S},) for all batch elements; got mismatches at "
                f"{bad[:5]}{' (truncated)' if len(bad) > 5 else ''}."
            )

        return (
            torch.stack(ys, dim=0),
            torch.stack(bases, dim=0),
            torch.stack(duals, dim=0),
        )


def solve_lp_dense(H, r, gamma):
    """
    Solve LP for single instance (not batched).

    H:     (S, SA) or (batch, S, SA) - if batched, solves first element
    r:     (SA,) or (batch, SA)
    gamma: (S,) or (batch, S)

    Returns:
        y_star: (SA,)
        basis:  (S,)
        dual:   (S,)  dual values for H y = gamma
    """
    if H.dim() == 3:
        H_np = H[0].detach().cpu().numpy()
        r_np = r[0].detach().cpu().numpy()
        gamma_np = gamma[0].detach().cpu().numpy()
    else:
        H_np = H.detach().cpu().numpy()
        r_np = r.detach().cpu().numpy()
        gamma_np = gamma.detach().cpu().numpy()
    return _solve_one_lp(H_np, r_np, gamma_np)


def solve_lp_gurobi(H, r, gamma):
    """Solve batched LP and return y_star, basis, and dual."""
    solver = DenseLPSolver()
    y_star, basis, dual = solver.solve(H, r, gamma)
    return y_star, basis, dual
