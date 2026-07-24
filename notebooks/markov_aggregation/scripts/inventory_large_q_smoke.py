from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mdp_lp.inventory_elasticity import create_inventory_from_lambda  # noqa: E402
from mdp_lp.soft_state_aggregation import SoftAggDense  # noqa: E402
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver  # noqa: E402
from mdp_lp.elasticity_model import ElasticityModel  # noqa: E402
from mdp_lp.train_elasticity import train_eval_elasticity  # noqa: E402
from mdp_lp.utils import solve_lp, set_seed  # noqa: E402
from cliff_aggregation_extensions import (  # noqa: E402
    StochasticSoftAggDense,
)


def _choose_basis_columns(H: np.ndarray, y: np.ndarray, n_rows: int) -> list[int]:
    """Choose a full-rank LP basis, preferring the induced policy basis.

    Large full-MDP oracles use policy iteration and bypass this generic helper.
    The SVD fallback is therefore retained for the small aggregated LPs, where
    its rank decisions are more reliable around stochastic near-degeneracy.
    """

    n_cols = H.shape[1]
    if n_cols % n_rows == 0:
        n_actions = n_cols // n_rows
        y_sa = np.asarray(y, dtype=np.float64).reshape(n_rows, n_actions)
        policy = np.argmax(y_sa, axis=1)
        policy_basis = (np.arange(n_rows) * n_actions + policy).astype(int).tolist()
        if int(np.linalg.matrix_rank(H[:, policy_basis], tol=1e-10)) == n_rows:
            return policy_basis

    preferred = list(np.flatnonzero(y > 1e-7))
    preferred_set = set(preferred)
    rest = [j for j in range(n_cols) if j not in preferred_set]
    chosen: list[int] = []
    current = np.empty((n_rows, 0), dtype=H.dtype)
    for j in preferred + rest:
        trial = np.column_stack([current, H[:, j]])
        if np.linalg.matrix_rank(trial, tol=1e-8) > len(chosen):
            chosen.append(j)
            current = trial
        if len(chosen) == n_rows:
            break

    if len(chosen) != n_rows:
        chosen = (preferred + rest)[:n_rows]
    return chosen


def _solve_discounted_mdp_from_h(
    H: np.ndarray,
    r: np.ndarray,
    gamma: np.ndarray,
    *,
    max_iterations: int = 500,
):
    """Solve a state-major discounted occupancy LP by policy iteration.

    Some Inventory actions have zero transition mass (a terminating/
    unavailable action), so the recovered transition kernel is allowed to be
    substochastic.  The final primal and dual residual checks are against the
    original H, not the recovered kernel.
    """

    n_states, n_cols = H.shape
    if n_cols % n_states:
        raise RuntimeError(
            f"Occupancy matrix width {n_cols} is not divisible by S={n_states}"
        )
    n_actions = n_cols // n_states
    column_sums = np.asarray(H.sum(axis=0), dtype=np.float64)
    beta = 1.0 - float(np.median(column_sums))
    if not 0.0 < beta < 1.0:
        raise RuntimeError(f"Could not infer a valid discount factor from H: {beta}")

    P = np.empty((n_states, n_actions, n_states), dtype=np.float64)
    for state in range(n_states):
        cols = slice(state * n_actions, (state + 1) * n_actions)
        block = -np.asarray(H[:, cols], dtype=np.float64).T
        block[:, state] += 1.0
        P[state] = block / beta
    if float(P.min()) < -1e-6:
        raise RuntimeError(
            f"Recovered transition matrix has negative probability {float(P.min()):.3e}"
        )
    P = np.clip(P, 0.0, None)
    if float(P.sum(axis=2).max()) > 1.0 + 1e-5:
        raise RuntimeError("Recovered transition matrix is not substochastic")

    rewards = np.asarray(r, dtype=np.float64).reshape(n_states, n_actions)
    gamma64 = np.asarray(gamma, dtype=np.float64)
    eye = np.eye(n_states, dtype=np.float64)
    states = np.arange(n_states)
    policy = np.argmax(rewards, axis=1)
    seen: set[bytes] = set()
    for _ in range(int(max_iterations)):
        P_pi = P[states, policy, :]
        value = np.linalg.solve(eye - beta * P_pi, rewards[states, policy])
        q_values = rewards + beta * np.matmul(P, value)
        improved = np.argmax(q_values, axis=1)
        if np.array_equal(improved, policy):
            break
        key = improved.tobytes()
        if key in seen:
            raise RuntimeError("Policy-iteration LP fallback cycled")
        seen.add(key)
        policy = improved
    else:
        raise RuntimeError("Policy-iteration LP fallback did not converge")

    basis = (states * n_actions + policy).astype(np.int64)
    H_basis = np.asarray(H[:, basis], dtype=np.float64)
    state_occupancy = np.linalg.solve(H_basis, gamma64)
    value = np.linalg.solve(H_basis.T, np.asarray(r, dtype=np.float64)[basis])
    if float(state_occupancy.min()) < -1e-6:
        raise RuntimeError(
            "Policy-iteration LP fallback produced negative occupancy "
            f"{float(state_occupancy.min()):.3e}"
        )
    occupancy = np.zeros((n_states, n_actions), dtype=np.float64)
    occupancy[states, policy] = state_occupancy
    y = occupancy.reshape(-1)
    flow_residual = float(np.max(np.abs(H @ y - gamma64)))
    dual_violation = float(np.max(np.asarray(r, dtype=np.float64) - H.T @ value))
    if flow_residual > 5e-5 or dual_violation > 5e-5:
        raise RuntimeError(
            "Policy-iteration LP fallback failed validation: "
            f"flow_residual={flow_residual:.3e}, dual_violation={dual_violation:.3e}"
        )
    print(
        "[LP policy retry] solved the substochastic occupancy LP "
        f"(flow_residual={flow_residual:.3e}, "
        f"dual_violation={dual_violation:.3e})",
        flush=True,
    )
    return (
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(basis, dtype=torch.long),
        torch.tensor(value, dtype=torch.float32),
    )


def _solve_one_lp_scipy(H_i, r_i, gamma_i, *, allow_policy_iteration: bool = True):
    from scipy.optimize import linprog
    from scipy import sparse

    S, SA = H_i.shape
    H_np = np.asarray(H_i, dtype=np.float64)
    r_np = np.asarray(r_i, dtype=np.float64)
    gamma_np = np.asarray(gamma_i, dtype=np.float64)
    bounds = [(0.0, None)] * SA
    dual_unscale = np.ones(S, dtype=np.float64)
    if allow_policy_iteration and SA >= 10_000:
        return _solve_discounted_mdp_from_h(H_np, r_np, gamma_np)
    res = linprog(
        c=-r_np,
        A_eq=sparse.csr_matrix(H_np),
        b_eq=gamma_np,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        first_message = res.message
        try:
            return _solve_discounted_mdp_from_h(H_np, r_np, gamma_np)
        except RuntimeError as policy_error:
            print(
                "[LP policy retry] structural solve was not applicable; "
                f"trying scaled HiGHS ({policy_error})",
                flush=True,
            )
        row_scale = np.maximum(np.max(np.abs(H_np), axis=1), np.abs(gamma_np))
        row_scale = np.maximum(row_scale, 1e-12)
        reward_scale = max(float(np.max(np.abs(r_np))), 1.0)
        H_scaled = H_np / row_scale[:, None]
        gamma_scaled = gamma_np / row_scale
        for method, options in (
            ("highs-ds", {"presolve": False}),
            ("highs-ipm", {"presolve": False}),
        ):
            retry = linprog(
                c=-r_np / reward_scale,
                A_eq=sparse.csr_matrix(H_scaled),
                b_eq=gamma_scaled,
                bounds=bounds,
                method=method,
                options=options,
            )
            if retry.success:
                residual = float(np.max(np.abs(H_np @ retry.x - gamma_np)))
                if residual <= 5e-5:
                    print(
                        "[LP retry] default HiGHS failed; solved the row-scaled "
                        f"LP with {method} (flow_residual={residual:.3e})",
                        flush=True,
                    )
                    res = retry
                    dual_unscale = reward_scale / row_scale
                    break
        else:
            raise RuntimeError(
                "SciPy/HiGHS failed to solve LP after scaled retries: "
                f"default={first_message}; last={retry.message}"
            )
    if not res.success:
        print(
            f"[LP retry] unexpected unsuccessful result: {res.message}", flush=True
        )
        raise RuntimeError(f"SciPy/HiGHS failed to solve LP: {res.message}")
    y = np.asarray(res.x, dtype=np.float32)
    basis = _choose_basis_columns(H_np, y, S)
    eqlin = getattr(res, "eqlin", None)
    if eqlin is not None and hasattr(eqlin, "marginals"):
        dual_np = -(
            np.asarray(eqlin.marginals, dtype=np.float64) * dual_unscale
        ).astype(np.float32)
    else:
        dual_np = np.zeros(S, dtype=np.float32)
    return (
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(basis, dtype=torch.long),
        torch.tensor(dual_np, dtype=torch.float32),
    )


def install_scipy_lp_backend(*, generic_constraints: bool = False) -> None:
    import mdp_lp.solvers.dense_lp_solver as dense
    import mdp_lp.solvers.differentiable_lp as diff

    if generic_constraints:
        def solve_one(H_i, r_i, gamma_i):
            return _solve_one_lp_scipy(
                H_i, r_i, gamma_i, allow_policy_iteration=False
            )
    else:
        solve_one = _solve_one_lp_scipy

    dense._solve_one_lp = solve_one

    def solve_lp_scipy(H, r, gamma):
        solver = dense.DenseLPSolver()
        return solver.solve(H, r, gamma)

    dense.solve_lp_gurobi = solve_lp_scipy
    diff.solve_lp_gurobi = solve_lp_scipy


def run_inprocess_elasticity(cmd_args: list[str], *, log_path: Path, scipy_lp: bool) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    old_argv = sys.argv[:]
    sys.argv = ["mdp_lp.run_elasticity_experiments", *cmd_args]
    try:
        with log_path.open("w", encoding="utf-8") as f, redirect_stdout(f), redirect_stderr(f):
            import mdp_lp.run_elasticity_experiments as exp

            if scipy_lp:
                install_scipy_lp_backend()
            exp.main()
        return 0
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception:
        import traceback

        with log_path.open("a", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        return 1
    finally:
        sys.argv = old_argv


TIERS = {
    "basic": {"b_max": 5, "s_max": 30, "q_max": 30, "n_agg": 12},
    "medium": {"b_max": 5, "s_max": 100, "q_max": 100, "n_agg": 12},
    "large": {"b_max": 10, "s_max": 210, "q_max": 250, "n_agg": 12},
}


def run_command(cmd: list[str], *, cwd: Path, log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, time.time() - t0


def train_result_path(root: Path, tier_name: str, method: str, seed: int, n_epochs: int) -> Path:
    tag = f"largeq_{tier_name}_{method}_seed{seed}"
    base = root / "exp_b" / "normal_margin"
    if method == "twostage":
        return base / "twostage" / f"tag_{tag}" / f"epochs_{n_epochs}" / "results.json"
    if method == "sa":
        return base / "dfl" / f"tag_{tag}" / f"n_agg_{TIERS[tier_name]['n_agg']}" / f"epochs_{n_epochs}" / "results.json"
    raise ValueError(method)


def make_train_args(
    *, tier_name: str, method: str, root: Path, seed: int, n_samples: int, n_epochs: int, pretrain_epochs: int
) -> Namespace:
    tier = TIERS[tier_name]
    save_dir = train_result_path(root, tier_name, method, seed, n_epochs).parent
    return Namespace(
        experiment_type="b",
        beta_0=3.5,
        beta_1=-1.0,
        beta_2=0.0,
        sigma_epsilon=0.3,
        p_constant=5.0,
        p_lognormal_mean=5.0,
        p_lognormal_std=0.6,
        p_related_lognormal_mean=8.0,
        p_related_lognormal_std=1.0,
        fixed_p=False,
        b_max=tier["b_max"],
        s_max=tier["s_max"],
        q_max=tier["q_max"],
        foc=0.0,
        voc=0.5,
        hc=2.0,
        bc=10.0,
        n_epochs=n_epochs,
        n_samples=n_samples,
        train_frac=0.25,
        algo="twostage" if method == "twostage" else "dfl",
        loss_weight=1.0 if method == "sa" else 0.0,
        lr=0.01,
        hidden_dim=32,
        beta=0.95,
        pretrain_epochs=pretrain_epochs,
        seed=seed,
        n_agg=-1 if method == "twostage" else tier["n_agg"],
        sa_tau=0.5,
        sa_r_embed=8,
        agg_fixed_W=False,
        agg_fixed_W_seed=1,
        qp_reg=10.0,
        qpth_kkt_tol=1e-3,
        qpth_grad_clip=1.0,
        enable_step_reject=False,
        step_reject_thresh=0.5,
        step_reject_max_consec=3,
        n_sketches=4,
        sketch_ratio=0.7,
        sketch_mode="gaussian",
        sketch_seed=None,
        n_perturb=4,
        sigma_perturb=0.3,
        perturb_mode="gaussian",
        perturb_seed=None,
        grad_clip=10.0,
        aug_lag_rho=0.1 if method == "sa" else 10000.0,
        policy_analysis=False,
        agg_lr_ratio=0.01,
        agg_warmup_epochs=30,
        fidelity_weight=0.0,
        normalize_disagg=True,
        agg_pretrain_epochs=0,
        agg_pretrain_oracle=True,
        agg_pretrain_lambda_aug=3,
        agg_pretrain_scale_min=0.3,
        agg_pretrain_scale_max=3.0,
        use_pi_eval_loss=False,
        requested_n_samples=n_samples,
        generated_n_samples=4 * n_samples,
        save_dir=save_dir,
    )


def prepare_elasticity_data_scipy(args: Namespace):
    set_seed(args.seed)
    total_n = 4 * int(args.n_samples)
    p = torch.exp(torch.randn(total_n) * args.p_lognormal_std + np.log(args.p_lognormal_mean))
    p_related = None
    gt_elasticity = ElasticityModel(
        beta_0=args.beta_0,
        beta_1=args.beta_1,
        beta_2=0.0,
        sigma_epsilon=args.sigma_epsilon,
    )
    lambda_true = gt_elasticity(p, p_related)

    def lp_solver(H, r, gamma):
        ys, bases, duals = [], [], []
        H_np = H.detach().cpu().numpy()
        r_np = r.detach().cpu().numpy()
        gamma_np = gamma.detach().cpu().numpy()
        for b in range(H_np.shape[0]):
            y, basis, dual = _solve_one_lp_scipy(H_np[b], r_np[b], gamma_np[b])
            ys.append(y)
            bases.append(basis)
            duals.append(dual)
        return torch.stack(ys), torch.stack(bases), torch.stack(duals)

    obj_true_list, y_true_list, dual_true_list = [], [], []
    for i in range(total_n):
        mdp = create_inventory_from_lambda(
            lambda_true[i : i + 1],
            args.b_max,
            args.s_max,
            args.q_max,
            args.foc,
            args.voc,
            args.hc,
            args.bc,
            args.beta,
            p[i : i + 1],
        )
        H = mdp.build_H(mdp.P)
        r_flat = mdp.r.reshape(1, -1)
        gamma = mdp.build_gamma(1, args.b_max + args.s_max + 1, torch.device("cpu"))
        gamma = torch.clamp(gamma, min=1e-8)
        gamma = gamma / gamma.sum(dim=1, keepdim=True)
        y_true, _, dual = lp_solver(H, r_flat, gamma)
        obj = torch.sum(r_flat * y_true, dim=1)
        obj_true_list.append(obj[0])
        y_true_list.append(y_true)
        dual_true_list.append(dual)

    obj_true = torch.stack(obj_true_list)
    y_true = torch.stack(y_true_list).squeeze(1)
    dual_true = torch.stack(dual_true_list).squeeze(1)
    feats = torch.stack([p, torch.randn(total_n) * 0.1], dim=1)
    dataset = TensorDataset(feats, obj_true, lambda_true, p, y_true, dual_true)
    train_size = total_n // 4
    test_size = total_n - train_size
    gen = torch.Generator().manual_seed(args.seed)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=gen)
    train_loader = DataLoader(train_dataset, batch_size=min(128, len(train_dataset)), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=min(128, len(test_dataset)), shuffle=False)
    return train_loader, test_loader, gt_elasticity


def run_direct_training_smoke(args: Namespace, *, log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        install_scipy_lp_backend()
        with log_path.open("w", encoding="utf-8") as f, redirect_stdout(f), redirect_stderr(f):
            train_loader, test_loader, gt_elasticity = prepare_elasticity_data_scipy(args)
            result = train_eval_elasticity(args, train_loader, test_loader, gt_elasticity)
            args.save_dir.mkdir(parents=True, exist_ok=True)

            def conv(x):
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().tolist()
                if isinstance(x, np.ndarray):
                    return x.tolist()
                if isinstance(x, Path):
                    return str(x)
                if isinstance(x, (np.integer, np.floating)):
                    return x.item()
                if isinstance(x, dict):
                    return {k: conv(v) for k, v in x.items()}
                if isinstance(x, list):
                    return [conv(v) for v in x]
                return x

            full = {**vars(args), **result, "status": "complete", "epoch": args.n_epochs}
            (args.save_dir / "results.json").write_text(json.dumps(conv(full), indent=2), encoding="utf-8")
        return 0, time.time() - t0
    except Exception:
        import traceback

        with log_path.open("a", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        return 1, time.time() - t0


def run_training_smoke(
    *,
    tier_name: str,
    method: str,
    root: Path,
    seed: int,
    n_samples: int,
    n_epochs: int,
    pretrain_epochs: int,
    force: bool,
    use_gpu: bool,
    scipy_lp: bool,
) -> dict:
    tier = TIERS[tier_name]
    result_path = train_result_path(root, tier_name, method, seed, n_epochs)
    if force and result_path.parent.exists():
        shutil.rmtree(result_path.parent)
    if result_path.exists() and not force:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        return {
            "kind": "training",
            "tier": tier_name,
            "method": method,
            "status": "skip",
            "result_path": str(result_path),
            "wall_time_sec": 0.0,
            "test_mean_regret": data.get("test_mean_regret"),
            "test_mean_oracle_reward": data.get("test_mean_oracle_reward"),
        }

    tag = f"largeq_{tier_name}_{method}_seed{seed}"
    cmd_args = [
        "--experiment_type",
        "b",
        "--cost_case",
        "normal_margin",
        "--bc",
        "10",
        "--n_samples",
        str(n_samples),
        "--pretrain_epochs",
        str(pretrain_epochs),
        "--b_max",
        str(tier["b_max"]),
        "--s_max",
        str(tier["s_max"]),
        "--q_max",
        str(tier["q_max"]),
        "--n_epochs",
        str(n_epochs),
        "--lr",
        "0.01",
        "--seed",
        str(seed),
        "--results_root",
        str(root),
        "--tag",
        tag,
    ]
    if method == "twostage":
        cmd_args.extend(["--algo", "twostage", "--n_agg", "-1", "--aug_lag_rho", "10000.0"])
    elif method == "sa":
        cmd_args.extend(
            [
                "--algo",
                "dfl",
                "--n_agg",
                str(tier["n_agg"]),
                "--sa_tau",
                "0.5",
                "--aug_lag_rho",
                "0.1",
                "--loss_weight",
                "1",
            ]
        )
    else:
        raise ValueError(method)
    if use_gpu:
        cmd_args.append("--use_gpu")

    if scipy_lp:
        train_args = make_train_args(
            tier_name=tier_name,
            method=method,
            root=root,
            seed=seed,
            n_samples=n_samples,
            n_epochs=n_epochs,
            pretrain_epochs=pretrain_epochs,
        )
        returncode, wall_time = run_direct_training_smoke(
            train_args,
            log_path=result_path.parent / "run.log",
        )
    else:
        t0 = time.time()
        returncode = run_inprocess_elasticity(
            cmd_args,
            log_path=result_path.parent / "run.log",
            scipy_lp=False,
        )
        wall_time = time.time() - t0
    record = {
        "kind": "training",
        "tier": tier_name,
        "method": method,
        "status": "ok" if returncode == 0 and result_path.exists() else f"fail:{returncode}",
        "result_path": str(result_path),
        "log_path": str(result_path.parent / "run.log"),
        "wall_time_sec": wall_time,
        "scipy_lp_backend": bool(scipy_lp),
    }
    if result_path.exists():
        data = json.loads(result_path.read_text(encoding="utf-8"))
        record.update(
            {
                "test_mean_regret": data.get("test_mean_regret"),
                "test_mean_oracle_reward": data.get("test_mean_oracle_reward"),
                "final_train_reward": (data.get("train_reward_per_epoch") or [None])[-1],
                "train_time_last_sec": (data.get("train_time") or [None])[-1],
                "solver_time_total_sec": sum(data.get("solver_time_sec_per_epoch") or []),
                "sa_time_total_sec": sum(data.get("sa_time_sec_per_epoch") or []),
            }
        )
    return record


def run_stochastic_layer_probe(
    *,
    tier_name: str,
    seed: int,
    n_membership_samples: int,
    noise_std: float,
) -> dict:
    """Cost probe for stochastic memberships on one generated Inventory MDP.

    This is intentionally not reported as a training result. The current
    Inventory training loop does not consume `info["samples"]`, while the Cliff
    training loop does. This probe estimates the per-batch cost that a proper
    stochastic Inventory training path would add at a given S*A scale.
    """
    torch.manual_seed(seed)
    tier = TIERS[tier_name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = tier["b_max"] + tier["s_max"] + 1
    A = tier["q_max"] + 1
    M = tier["n_agg"]
    # InventoryLP currently builds sparse transition tensors with CPU indices,
    # so construct the MDP on CPU first and then move dense tensors to CUDA for
    # the aggregation/LP timing probe.
    lam = torch.tensor([12.0])
    price = torch.tensor([10.0])
    mdp = create_inventory_from_lambda(
        lam,
        tier["b_max"],
        tier["s_max"],
        tier["q_max"],
        0.0,
        0.5,
        2.0,
        10.0,
        0.95,
        price,
    )
    mdp.P = mdp.P.to(device)
    mdp.r = mdp.r.to(device)
    gamma = mdp.build_gamma(1, S, device)
    agg = StochasticSoftAggDense(
        n_states=S,
        n_actions=A,
        n_clusters=M,
        r_embed=8,
        n_membership_samples=n_membership_samples,
        noise_std=noise_std,
    ).to(device)
    if hasattr(torch.cuda, "synchronize") and device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    P_agg, r_agg, gamma_agg, info = agg(mdp.P, mdp.r, gamma, tau=0.5)
    H = mdp.build_H(P_agg)
    r_flat = r_agg.reshape(1, -1)
    y_base = DifferentiableLPSolver.apply(H, r_flat, gamma_agg)
    sample_ys = []
    for P_i, r_i, gamma_i, _ in info["samples"]:
        H_i = mdp.build_H(P_i)
        sample_ys.append(DifferentiableLPSolver.apply(H_i, r_i.reshape(1, -1), gamma_i))
    y_mean = torch.stack([y_base, *sample_ys], dim=0).mean(dim=0)
    if hasattr(torch.cuda, "synchronize") and device.type == "cuda":
        torch.cuda.synchronize()
    wall_time = time.time() - t0
    return {
        "kind": "stochastic_layer_probe",
        "tier": tier_name,
        "method": "stochastic_layer",
        "status": "ok",
        "device": str(device),
        "S": S,
        "A": A,
        "SA": S * A,
        "M": M,
        "n_membership_samples": n_membership_samples,
        "noise_std": noise_std,
        "wall_time_sec": wall_time,
        "agg_reported_runtime_sec": float(info.get("runtime_sec", 0.0)),
        "y_mean_sum": float(y_mean.detach().sum().cpu()),
        "note": (
            "Forward-only cost probe. Inventory training loop still needs "
            "sample-averaging plumbing, and DifferentiableLPSolver backward is "
            "not CUDA-safe because basis indices are returned on CPU."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", nargs="+", default=["large"], choices=list(TIERS))
    parser.add_argument("--methods", nargs="+", default=["twostage", "sa", "stochastic_layer"])
    parser.add_argument("--root", type=Path, default=REPO / "notebooks/reproduction/repro/results/inventory_large_q_smoke")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=2)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--pretrain-epochs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--scipy-lp", action="store_true", default=True)
    parser.add_argument("--no-scipy-lp", dest="scipy_lp", action="store_false")
    parser.add_argument("--stoch-samples", type=int, default=2)
    parser.add_argument("--stoch-noise", type=float, default=0.05)
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    records = []
    for tier_name in args.tiers:
        for method in args.methods:
            print(f"[smoke] tier={tier_name} method={method}", flush=True)
            if method in ("twostage", "sa"):
                record = run_training_smoke(
                    tier_name=tier_name,
                    method=method,
                    root=args.root,
                    seed=args.seed,
                    n_samples=args.n_samples,
                    n_epochs=args.n_epochs,
                    pretrain_epochs=args.pretrain_epochs,
                    force=args.force,
                    use_gpu=args.use_gpu,
                    scipy_lp=args.scipy_lp,
                )
            elif method == "stochastic_layer":
                record = run_stochastic_layer_probe(
                    tier_name=tier_name,
                    seed=args.seed,
                    n_membership_samples=args.stoch_samples,
                    noise_std=args.stoch_noise,
                )
            else:
                raise ValueError(method)
            records.append(record)
            print(json.dumps(record, indent=2), flush=True)

    summary_path = args.root / "summary.json"
    summary_path.write_text(json.dumps({"records": records, "tiers": TIERS}, indent=2), encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
