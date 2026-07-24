from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from mdp_lp.elasticity_model import ElasticityModel  # noqa: E402
from mdp_lp.inventory_elasticity import create_inventory_from_lambda  # noqa: E402
from mdp_lp.soft_state_aggregation import SoftAggDense  # noqa: E402
import mdp_lp.train_elasticity as train_elasticity_mod  # noqa: E402
from mdp_lp.train_elasticity import train_eval_elasticity  # noqa: E402
from mdp_lp.utils import set_seed  # noqa: E402
from cliff_aggregation_extensions import (  # noqa: E402
    StochasticSoftAggDense,
)
from inventory_large_q_smoke import (  # noqa: E402
    TIERS,
    _solve_one_lp_scipy,
    install_scipy_lp_backend,
)


METHODS = {
    "ts": {"algo": "twostage", "n_agg": -1, "aug_lag_rho": 10000.0, "loss_weight": 0.0},
    "dfl": {"algo": "dfl", "n_agg": -1, "aug_lag_rho": 10000.0, "loss_weight": 1.0},
    "qp": {"algo": "dfl_qpth", "n_agg": -1, "aug_lag_rho": 10000.0, "loss_weight": 1.0},
    "qp_struct": {
        "algo": "dfl_qpth",
        "n_agg": -1,
        "aug_lag_rho": 10000.0,
        "loss_weight": 1.0,
        "qp_backend": "structured",
    },
    "sketch": {"algo": "dfl_sketch", "n_agg": -1, "aug_lag_rho": 0.1, "loss_weight": 1.0},
    "sa": {"algo": "dfl", "n_agg": None, "aug_lag_rho": 0.1, "loss_weight": 1.0},
    "stoch_sa": {"algo": "dfl", "n_agg": None, "aug_lag_rho": 0.1, "loss_weight": 1.0},
}


class AveragedStochasticSoftAggDense(StochasticSoftAggDense):
    """Inventory-compatible stochastic aggregation.

    `train_elasticity.py` does not consume `info["samples"]`. For speed and
    robustness smoke tests, make stochasticity active by averaging the sampled
    aggregated MDP tensors before the LP layer.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("fixed_W", None)
        kwargs.pop("fixed_W_seed", None)
        super().__init__(*args, **kwargs)

    @staticmethod
    def disaggregate_y(y_agg, W, n_actions):
        return SoftAggDense.disaggregate_y(y_agg, W, n_actions)

    def forward(self, P, r, gamma, tau=0.5):
        P_base, r_base, gamma_base, info = super().forward(P, r, gamma, tau=tau)
        samples = info.get("samples", [])
        if samples:
            P_all = [P_base] + [x[0] for x in samples]
            r_all = [r_base] + [x[1] for x in samples]
            gamma_all = [gamma_base] + [x[2] for x in samples]
            P_base = torch.stack(P_all, dim=0).mean(dim=0)
            r_base = torch.stack(r_all, dim=0).mean(dim=0)
            gamma_base = torch.stack(gamma_all, dim=0).mean(dim=0)
        return P_base, r_base, gamma_base, info


def _jsonable(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    return x


def solve_discounted_mdp_policy(
    P: torch.Tensor,
    r: torch.Tensor,
    gamma: torch.Tensor,
    *,
    beta: float,
    n_threads: int = 8,
    max_iterations: int = 500,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Solve the discounted occupancy LP through exact policy iteration.

    For a fixed deterministic policy, its occupancy basis is
    ``I - beta * P_pi.T``.  Discounting makes this matrix nonsingular, so policy
    iteration recovers the same optimal extreme point, dual value vector, and
    basis that the full occupancy LP would provide without sending 55,471
    variables through a generic LP solver.
    """

    torch.set_num_threads(max(1, int(n_threads)))
    P64 = P.detach().to(device="cpu", dtype=torch.float64)
    r64 = r.detach().to(device="cpu", dtype=torch.float64)
    gamma64 = gamma.detach().to(device="cpu", dtype=torch.float64)
    n_states, n_actions, _ = P64.shape
    states = torch.arange(n_states, dtype=torch.long)
    eye = torch.eye(n_states, dtype=torch.float64)
    policy = torch.argmax(r64, dim=1)
    seen: set[bytes] = set()

    for iteration in range(1, int(max_iterations) + 1):
        P_pi = P64[states, policy, :]
        r_pi = r64[states, policy]
        value = torch.linalg.solve(eye - float(beta) * P_pi, r_pi)
        q_values = r64 + float(beta) * torch.matmul(P64, value)
        improved = torch.argmax(q_values, dim=1)
        if torch.equal(improved, policy):
            break
        key = improved.numpy().tobytes()
        if key in seen:
            raise RuntimeError(
                f"Policy iteration cycled after {iteration} iterations"
            )
        seen.add(key)
        policy = improved
    else:
        raise RuntimeError(
            f"Policy iteration did not converge in {max_iterations} iterations"
        )

    P_pi = P64[states, policy, :]
    r_pi = r64[states, policy]
    value = torch.linalg.solve(eye - float(beta) * P_pi, r_pi)
    occupancy_state = torch.linalg.solve(
        eye - float(beta) * P_pi.T,
        gamma64,
    )
    if float(occupancy_state.min().item()) < -1e-8:
        raise RuntimeError(
            "Policy occupancy contains a materially negative state mass: "
            f"{float(occupancy_state.min().item()):.3e}"
        )

    occupancy = torch.zeros((n_states, n_actions), dtype=torch.float64)
    occupancy[states, policy] = occupancy_state
    basis = states * n_actions + policy

    q_values = r64 + float(beta) * torch.matmul(P64, value)
    bellman_residual = float(
        torch.max(torch.abs(value - torch.max(q_values, dim=1).values)).item()
    )
    flow_residual = float(
        torch.max(
            torch.abs(
                occupancy_state
                - float(beta) * P_pi.T.matmul(occupancy_state)
                - gamma64
            )
        ).item()
    )
    primal_obj = float(torch.sum(r64 * occupancy).item())
    dual_obj = float(torch.dot(gamma64, value).item())
    diagnostics = {
        "backend": "policy_iteration",
        "policy_iterations": iteration,
        "bellman_residual_inf": bellman_residual,
        "flow_residual_inf": flow_residual,
        "primal_obj_float64": primal_obj,
        "dual_obj_float64": dual_obj,
        "primal_dual_gap": dual_obj - primal_obj,
        "oracle_threads": int(n_threads),
    }
    return (
        occupancy.reshape(-1).to(dtype=torch.float32),
        basis.to(dtype=torch.long),
        value.to(dtype=torch.float32),
        diagnostics,
    )


def oracle_worker_main(args: argparse.Namespace) -> int:
    torch.set_num_threads(max(1, int(getattr(args, "oracle_threads", 8))))
    b_max, s_max, q_max = args.b_max, args.s_max, args.q_max
    lam = torch.tensor([float(args.lam)], dtype=torch.float32)
    price = torch.tensor([float(args.price)], dtype=torch.float32)
    t0 = time.time()
    mdp = create_inventory_from_lambda(
        lam,
        b_max,
        s_max,
        q_max,
        float(args.foc),
        float(args.voc),
        float(args.hc),
        float(args.bc),
        float(args.beta),
        price,
    )
    build_mdp_sec = time.time() - t0
    gamma = mdp.build_gamma(1, b_max + s_max + 1, torch.device("cpu"))
    gamma = torch.clamp(gamma, min=1e-8)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)
    r_flat = mdp.r.reshape(1, -1)
    oracle_backend = getattr(args, "oracle_backend", "policy")

    build_h_sec = 0.0
    t0 = time.time()
    if oracle_backend == "scipy":
        H = mdp.build_H(mdp.P)
        build_h_sec = time.time() - t0
        t0 = time.time()
        y, basis, dual = _solve_one_lp_scipy(
            H[0].detach().numpy(),
            r_flat[0].detach().numpy(),
            gamma[0].detach().numpy(),
        )
        diagnostics = {"backend": "scipy_highs"}
    elif oracle_backend == "policy":
        y, basis, dual, diagnostics = solve_discounted_mdp_policy(
            mdp.P[0],
            mdp.r[0],
            gamma[0],
            beta=float(args.beta),
            n_threads=int(getattr(args, "oracle_threads", 8)),
        )
    else:
        raise ValueError(f"Unsupported oracle backend: {oracle_backend}")
    solve_sec = time.time() - t0
    obj = float(torch.sum(r_flat[0] * y).item())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "obj_true": obj,
            "y_true": y,
            "dual_true": dual,
            "basis": basis,
            "timing": {
                "build_mdp_sec": build_mdp_sec,
                "build_h_sec": build_h_sec,
                "oracle_lp_sec": solve_sec,
                "total_sec": build_mdp_sec + build_h_sec + solve_sec,
                **diagnostics,
            },
            "config": {
                "b_max": b_max,
                "s_max": s_max,
                "q_max": q_max,
                "lambda": float(args.lam),
                "price": float(args.price),
                "oracle_backend": oracle_backend,
            },
        },
        out,
    )
    return 0


def make_base_args(
    *,
    tier_name: str,
    method: str,
    root: Path,
    seed: int,
    n_samples: int,
    n_epochs: int,
    pretrain_epochs: int,
) -> Namespace:
    tier = TIERS[tier_name]
    method_cfg = METHODS[method]
    n_agg = tier["n_agg"] if method_cfg["n_agg"] is None else method_cfg["n_agg"]
    save_dir = root / "runs" / tier_name / method / f"seed_{seed}_ne{n_epochs}_ns{n_samples}"
    return Namespace(
        experiment_type="b",
        beta_0=3.5,
        beta_1=-1.0,
        beta_2=0.0,
        sigma_epsilon=0.3,
        p_lognormal_mean=5.0,
        p_lognormal_std=0.6,
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
        algo=method_cfg["algo"],
        loss_weight=method_cfg["loss_weight"],
        lr=0.01,
        hidden_dim=32,
        beta=0.95,
        pretrain_epochs=pretrain_epochs,
        seed=seed,
        n_agg=n_agg,
        sa_tau=0.5,
        sa_r_embed=8,
        agg_fixed_W=False,
        agg_fixed_W_seed=1,
        qp_reg=10.0,
        qp_backend=method_cfg.get("qp_backend", "qpth"),
        structured_qp_tol=1e-7,
        structured_qp_max_iter=500,
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
        aug_lag_rho=method_cfg["aug_lag_rho"],
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
        stochastic_membership_samples=2,
        stochastic_membership_noise=0.05,
        use_cached_eval_oracle=True,
        use_aggregated_eval_policy=True,
        skip_initial_reward_eval=True,
        skip_gradient_diagnostics=True,
        oracle_backend="policy",
        oracle_threads=8,
        training_threads=64,
        # Evaluation solves an S-by-S true-MDP system for every sample.  Keep
        # this independent of the training batch so a large test split does
        # not materialize one enormous B x S x S evaluation tensor.  It does
        # not change any per-sample policy/value or training gradient.
        eval_batch_size=1,
    )


@contextmanager
def patched_stochastic_agg(method: str, args: Namespace):
    old = train_elasticity_mod.SoftAggDense
    try:
        if method == "stoch_sa":
            class ConfiguredAveragedStochasticSoftAggDense(AveragedStochasticSoftAggDense):
                def __init__(self, *ctor_args, **ctor_kwargs):
                    ctor_kwargs["n_membership_samples"] = int(
                        getattr(args, "stochastic_membership_samples", 2)
                    )
                    ctor_kwargs["noise_std"] = float(
                        getattr(args, "stochastic_membership_noise", 0.05)
                    )
                    super().__init__(*ctor_args, **ctor_kwargs)

            train_elasticity_mod.SoftAggDense = ConfiguredAveragedStochasticSoftAggDense
        yield
    finally:
        train_elasticity_mod.SoftAggDense = old


def sample_data(args: Namespace):
    set_seed(args.seed)
    total_n = 4 * int(args.n_samples)
    p = torch.exp(torch.randn(total_n) * args.p_lognormal_std + math.log(args.p_lognormal_mean))
    gt = ElasticityModel(
        beta_0=args.beta_0,
        beta_1=args.beta_1,
        beta_2=0.0,
        sigma_epsilon=args.sigma_epsilon,
    )
    lam = gt(p, None)
    feats = torch.stack([p, torch.randn(total_n) * 0.1], dim=1)
    return feats, p, lam, gt


def run_oracle_sample(
    *,
    script: Path,
    out: Path,
    args: Namespace,
    lam: float,
    price: float,
    force: bool,
) -> dict:
    if out.exists() and not force:
        payload = torch.load(out, map_location="cpu", weights_only=False)
        timing = payload.get("timing", {})
        return {"status": "skip", "path": str(out), **timing}
    cmd = [
        sys.executable,
        str(script),
        "--oracle-worker",
        "--out",
        str(out),
        "--b-max",
        str(args.b_max),
        "--s-max",
        str(args.s_max),
        "--q-max",
        str(args.q_max),
        "--lam",
        str(lam),
        "--price",
        str(price),
        "--foc",
        str(args.foc),
        "--voc",
        str(args.voc),
        "--hc",
        str(args.hc),
        "--bc",
        str(args.bc),
        "--beta",
        str(args.beta),
        "--oracle-backend",
        str(getattr(args, "oracle_backend", "policy")),
        "--oracle-threads",
        str(getattr(args, "oracle_threads", 8)),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall = time.time() - t0
    if not out.exists():
        log = out.with_suffix(".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(proc.stdout, encoding="utf-8", errors="replace")
        return {"status": f"fail:{proc.returncode}", "path": str(out), "log": str(log), "wall_sec": wall}
    payload = torch.load(out, map_location="cpu", weights_only=False)
    status = "ok" if proc.returncode == 0 else f"ok_cache_worker_exit_{proc.returncode}"
    return {
        "status": status,
        "path": str(out),
        "wall_sec": wall,
        "worker_returncode": proc.returncode,
        **payload.get("timing", {}),
    }


def prepare_cached_dataset(
    *,
    root: Path,
    tier_name: str,
    method: str,
    args: Namespace,
    force_oracle: bool,
    oracle_workers: int,
):
    feats, prices, lambdas, gt = sample_data(args)
    oracle_dir = root / "oracle_cache" / tier_name / f"seed_{args.seed}_ns{args.n_samples}"
    script = Path(__file__).resolve()
    jobs = []
    for i, (lam, price) in enumerate(zip(lambdas.tolist(), prices.tolist())):
        out = oracle_dir / f"sample_{i:04d}.pt"
        jobs.append((i, out, float(lam), float(price)))

    t0 = time.time()
    oracle_records = []
    with ThreadPoolExecutor(max_workers=max(1, int(oracle_workers))) as ex:
        futs = [
            ex.submit(
                run_oracle_sample,
                script=script,
                out=out,
                args=args,
                lam=lam,
                price=price,
                force=force_oracle,
            )
            for _, out, lam, price in jobs
        ]
        for fut in as_completed(futs):
            oracle_records.append(fut.result())
    oracle_wall = time.time() - t0
    failed = [r for r in oracle_records if not str(r["status"]).startswith(("ok", "skip"))]
    if failed:
        raise RuntimeError(f"Oracle cache failed: {failed[:2]}")

    obj, y, dual = [], [], []
    for _, out, _, _ in jobs:
        payload = torch.load(out, map_location="cpu", weights_only=False)
        obj.append(float(payload["obj_true"]))
        y.append(payload["y_true"])
        dual.append(payload["dual_true"])
    dataset = TensorDataset(
        feats,
        torch.tensor(obj, dtype=torch.float32),
        lambdas,
        prices,
        torch.stack(y),
        torch.stack(dual),
    )
    train_size = len(dataset) // 4
    test_size = len(dataset) - train_size
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=gen)
    requested_train_batch = int(getattr(args, "train_batch_size", 128))
    train_loader = DataLoader(
        train_ds,
        batch_size=min(max(1, requested_train_batch), len(train_ds)),
        shuffle=True,
    )
    test_loader = DataLoader(test_ds, batch_size=min(128, len(test_ds)), shuffle=False)
    return train_loader, test_loader, gt, {
        "oracle_wall_sec": oracle_wall,
        "oracle_records": oracle_records,
        "test_indices": list(test_ds.indices),
        "oracle_obj_values": obj,
    }


def run_one(args: Namespace, *, root: Path, tier_name: str, method: str, force: bool, force_oracle: bool, oracle_workers: int) -> dict:
    result_path = args.save_dir / "results.json"
    if result_path.exists() and not force:
        data = json.loads(result_path.read_text(encoding="utf-8"))
        return {"status": "skip", "tier": tier_name, "method": method, "result_path": str(result_path), **data.get("timing_summary", {})}

    args.save_dir.mkdir(parents=True, exist_ok=True)
    # The SciPy fallback recognizes a full discounted-MDP flow matrix and
    # accelerates the regular occupancy LP. A row-sketched LP deliberately
    # has fewer, mixed constraints, so it is no longer an S x (S*A) flow
    # system. Sketch therefore uses the generic HiGHS branch, while the
    # remaining methods may use the structural policy-iteration shortcut.
    install_scipy_lp_backend(generic_constraints=(method == "sketch"))
    t0 = time.time()
    train_loader, test_loader, gt, cache_info = prepare_cached_dataset(
        root=root,
        tier_name=tier_name,
        method=method,
        args=args,
        force_oracle=force_oracle,
        oracle_workers=oracle_workers,
    )
    data_sec = time.time() - t0
    torch.set_num_threads(max(1, int(getattr(args, "training_threads", 64))))
    t0 = time.time()
    try:
        eval_batch_size = max(1, int(getattr(args, "eval_batch_size", 1)))
        evaluation_loader = DataLoader(
            test_loader.dataset,
            batch_size=min(eval_batch_size, len(test_loader.dataset)),
            shuffle=False,
        )
        with patched_stochastic_agg(method, args):
            result = train_eval_elasticity(args, train_loader, evaluation_loader, gt)
            train_eval_sec = time.time() - t0
    except Exception as exc:
        train_eval_sec = time.time() - t0
        test_indices = cache_info.get("test_indices", [])
        oracle_values = cache_info.get("oracle_obj_values", [])
        fallback_oracle = None
        if test_indices and oracle_values:
            fallback_oracle = float(np.mean([oracle_values[i] for i in test_indices]))
        timing_summary = {
            "data_cache_total_sec": data_sec,
            "oracle_wall_sec": cache_info["oracle_wall_sec"],
            "train_eval_sec_until_failure": train_eval_sec,
            "total_sec_until_failure": data_sec + train_eval_sec,
            "n_oracle_records": len(cache_info["oracle_records"]),
            "cached_test_mean_oracle_reward": fallback_oracle,
        }
        fail = {
            "status": "fail_train_eval",
            "tier": tier_name,
            "method": method,
            "result_path": str(result_path),
            "error": repr(exc),
            "note": "Training reached this point, but train_eval_elasticity failed during repeated full-LP evaluation. cached_test_mean_oracle_reward is computed from the oracle cache.",
            **timing_summary,
        }
        (args.save_dir / "failed_speed_record.json").write_text(
            json.dumps(_jsonable(fail), indent=2),
            encoding="utf-8",
        )
        return fail
    total_sec = data_sec + train_eval_sec
    timing_summary = {
        "data_cache_total_sec": data_sec,
        "oracle_wall_sec": cache_info["oracle_wall_sec"],
        "train_eval_sec": train_eval_sec,
        "total_sec": total_sec,
        "solver_time_total_sec": float(sum(result.get("solver_time_sec_per_epoch", []) or [])),
        "sa_time_total_sec": float(sum(result.get("sa_time_sec_per_epoch", []) or [])),
        "n_oracle_records": len(cache_info["oracle_records"]),
    }
    full = {
        **vars(args),
        **result,
        "status": "complete",
        "timing_summary": timing_summary,
        "tier": tier_name,
        "method": method,
    }
    result_path.write_text(json.dumps(_jsonable(full), indent=2), encoding="utf-8")
    return {
        "status": "ok",
        "tier": tier_name,
        "method": method,
        "result_path": str(result_path),
        "test_mean_regret": result.get("test_mean_regret"),
        "test_mean_oracle_reward": result.get("test_mean_oracle_reward"),
        **timing_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-worker", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--b-max", type=int)
    parser.add_argument("--s-max", type=int)
    parser.add_argument("--q-max", type=int)
    parser.add_argument("--lam", type=float)
    parser.add_argument("--price", type=float)
    parser.add_argument("--foc", type=float, default=0.0)
    parser.add_argument("--voc", type=float, default=0.5)
    parser.add_argument("--hc", type=float, default=2.0)
    parser.add_argument("--bc", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--oracle-backend", choices=["policy", "scipy"], default="policy")
    parser.add_argument("--oracle-threads", type=int, default=8)

    parser.add_argument("--tiers", nargs="+", default=["basic"])
    parser.add_argument("--methods", nargs="+", default=["sa"])
    parser.add_argument("--root", type=Path, default=REPO / "notebooks/reproduction/repro/results/inventory_large_q_speed")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--oracle-workers", type=int, default=1)
    parser.add_argument("--training-threads", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-oracle", action="store_true")
    args = parser.parse_args()

    if args.oracle_worker:
        return oracle_worker_main(args)

    args.root.mkdir(parents=True, exist_ok=True)
    rows = []
    for tier_name in args.tiers:
        for method in args.methods:
            train_args = make_base_args(
                tier_name=tier_name,
                method=method,
                root=args.root,
                seed=args.seed,
                n_samples=args.n_samples,
                n_epochs=args.n_epochs,
                pretrain_epochs=args.pretrain_epochs,
            )
            train_args.oracle_backend = args.oracle_backend
            train_args.oracle_threads = args.oracle_threads
            train_args.training_threads = args.training_threads
            print(f"[speed] tier={tier_name} method={method}", flush=True)
            row = run_one(
                train_args,
                root=args.root,
                tier_name=tier_name,
                method=method,
                force=args.force,
                force_oracle=args.force_oracle,
                oracle_workers=args.oracle_workers,
            )
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)
    summary = {
        "tiers": TIERS,
        "n_samples_arg": args.n_samples,
        "n_epochs": args.n_epochs,
        "pretrain_epochs": args.pretrain_epochs,
        "oracle_workers": args.oracle_workers,
        "rows": rows,
    }
    out = args.root / "speed_summary.json"
    out.write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
