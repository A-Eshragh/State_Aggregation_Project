"""Toy numerical demo: stochastic random matrices smooth an LP's gradient.

We sweep a 1-D parameter theta through a small parametric LP

    maximize  c(theta)^T x
    subject to A x <= b,  0 <= x <= x_max

with the cost ROTATED through cost space:

    c(theta) = cos(theta) * c0 + sin(theta) * c1,

so as theta sweeps [0, 2*pi] the cost direction makes one full revolution
and the LP optimum hops through every vertex its normal-fan crosses. The
induced loss L(theta) = c(theta)^T x*(theta) is piecewise-affine in cos/sin
of theta with many kinks; its gradient dL/dtheta (finite-difference) is
piecewise-constant with sharp jumps at every kink.

We compare three solvers along the same theta grid:

    1) plain LP                                   -- spiky landscape
    2) sketched LP via cost-side perturbation     -- smooth landscape
    3) sketched LP via constraint-side row sketch -- smooth landscape

For each ensemble member m = 1..K, we either
    (cost)        solve max (c(theta) + sigma * eps_m)^T x  s.t.  A x <= b
    (constraint)  solve max c(theta)^T x  s.t. (S_m A) x <= (S_m b)
and then return the ensemble mean of the K optimal x's. Loss at theta is
evaluated as c(theta)^T x_avg(theta) (NOT the perturbed cost) so all curves
live on a comparable y-axis.

The figure shows that the AVERAGED LP solution is a smooth function of theta
even though every individual sketch is itself piecewise-constant.

Usage:
    python -m mdp_lp.sketch_smoothing_demo                # default K, sigma
    python -m mdp_lp.sketch_smoothing_demo --K 32         # bigger ensemble
    python -m mdp_lp.sketch_smoothing_demo --out ./toy.pdf
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


# ---------------------------------------------------------------- LP family


def make_lp(seed: int = 0, n: int = 12, m: int = 5,
            cost_seed: int = 7) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Build a random parametric LP with a ROTATING cost direction.

    Returns (A, b, c0, c1, x_upper) defining

        max  (cos(theta) * c0 + sin(theta) * c1)^T x
        s.t. A x <= b,  0 <= x <= x_upper

    A and b are drawn so the polytope is bounded and has many vertices.
    c0 and c1 are orthonormalised so the cost direction sweeps a unit
    circle in the c0-c1 plane as theta ranges over [0, 2*pi].
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.2, 1.0, size=(m, n))
    b = rng.uniform(1.0, 2.0, size=m)
    crng = np.random.default_rng(cost_seed)
    c0 = crng.standard_normal(n)
    c0 = c0 / np.linalg.norm(c0)
    c1 = crng.standard_normal(n)
    c1 = c1 - (c1 @ c0) * c0
    c1 = c1 / np.linalg.norm(c1)
    x_upper = 5.0
    return A, b, c0, c1, x_upper


def cost_at(theta: float, c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """Rotated unit-norm cost direction in the c0-c1 plane."""
    return float(np.cos(theta)) * c0 + float(np.sin(theta)) * c1


def solve_lp(c: np.ndarray, A: np.ndarray, b: np.ndarray, x_upper: float) -> np.ndarray:
    """Solve  max c^T x  s.t.  A x <= b,  0 <= x <= x_upper  via HiGHS.

    Returns x*. If infeasible/unbounded (can happen under aggressive
    constraint sketches), returns NaN.
    """
    n = A.shape[1]
    res = linprog(
        -c, A_ub=A, b_ub=b,
        bounds=[(0.0, x_upper)] * n,
        method="highs",
    )
    if not res.success or res.x is None:
        return np.full(n, np.nan)
    return np.asarray(res.x, dtype=float)


# ---------------------------------------------------------------- solvers


def vanilla_sweep(thetas: np.ndarray, A, b, c0, c1, x_upper):
    n = A.shape[1]
    Y = np.zeros((len(thetas), n))
    L = np.zeros(len(thetas))
    for i, th in enumerate(thetas):
        c = cost_at(th, c0, c1)
        x = solve_lp(c, A, b, x_upper)
        Y[i] = x
        L[i] = float(c @ x)
    return L, Y


def cost_sketch_sweep(thetas, A, b, c0, c1, x_upper,
                      K: int = 16, sigma: float = 0.4, seed: int = 0):
    """Cost-side perturbation: add Gaussian noise sigma*eps_m to c(theta).

    The K eps_m are sampled ONCE and reused for every theta, so the resulting
    L_avg(theta) is a deterministic function of theta (no Monte-Carlo
    noise across the sweep).
    """
    rng = np.random.default_rng(seed)
    n = A.shape[1]
    eps = rng.standard_normal((K, n))
    Y_avg = np.zeros((len(thetas), n))
    L_avg = np.zeros(len(thetas))
    for i, th in enumerate(thetas):
        c = cost_at(th, c0, c1)
        accum = np.zeros(n)
        valid = 0
        for m in range(K):
            x = solve_lp(c + sigma * eps[m], A, b, x_upper)
            if np.all(np.isfinite(x)):
                accum += x
                valid += 1
        Y_avg[i] = accum / max(valid, 1)
        L_avg[i] = float(c @ Y_avg[i])
    return L_avg, Y_avg


def constraint_sketch_sweep(thetas, A, b, c0, c1, x_upper,
                            K: int = 16, k_ratio: float = 0.8, seed: int = 0):
    """Row-sketched constraints: (S_m A) x <= (S_m b) with k = round(k_ratio * m).

    S_m are pre-sampled and held fixed across the theta sweep. The 0 <= x <=
    x_upper box keeps each sketched LP bounded even though SH y = S gamma is
    a relaxation of the original constraint.
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape
    k = max(1, int(round(k_ratio * m)))
    S_stack = rng.standard_normal((K, k, m)) / np.sqrt(k)
    Y_avg = np.zeros((len(thetas), n))
    L_avg = np.zeros(len(thetas))
    for i, th in enumerate(thetas):
        c = cost_at(th, c0, c1)
        accum = np.zeros(n)
        valid = 0
        for s in range(K):
            S = S_stack[s]
            x = solve_lp(c, S @ A, S @ b, x_upper)
            if np.all(np.isfinite(x)):
                accum += x
                valid += 1
        Y_avg[i] = accum / max(valid, 1)
        L_avg[i] = float(c @ Y_avg[i])
    return L_avg, Y_avg


# ---------------------------------------------------------------- helpers


def grad_finite_diff(L: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return np.gradient(L, theta)


def grad_implicit(thetas: np.ndarray, Y: np.ndarray,
                  c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    """Implicit / envelope-theorem gradient: g(θ) = y(θ)^T (dc/dθ).

    For c(θ) = cosθ c0 + sinθ c1 we have dc/dθ = -sinθ c0 + cosθ c1.
    This is what an autograd LP layer returns when the basis is held
    fixed (the "y-times-dc-by-dθ" sensitivity), and is the natural
    analogue of `dL/dθ` an SGD-trained predictor would receive.
    """
    s = np.sin(thetas); c = np.cos(thetas)
    dc = (-np.outer(s, c0) + np.outer(c, c1))     # (T, n)
    return np.einsum("ti,ti->t", Y, dc)


def roughness(x: np.ndarray) -> float:
    """Total variation of x: sum of |Δ| over consecutive elements."""
    return float(np.abs(np.diff(x)).sum())


# ---------------------------------------------------------------- plotting


def make_figure(thetas, L_v, L_cs, L_xs, K: int, sigma: float, k_ratio: float,
                x_idx: int, Y_v, Y_cs, Y_xs, c0, c1, out_path: str):
    # Implicit / envelope gradient g = y^T (dc/dθ): the natural autograd
    # analogue of dL/dθ for an LP layer with held-fixed basis. This is the
    # signal an SGD predictor sees in DFL training.
    g_v = grad_implicit(thetas, Y_v, c0, c1)
    g_cs = grad_implicit(thetas, Y_cs, c0, c1)
    g_xs = grad_implicit(thetas, Y_xs, c0, c1)

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.5), sharex=True,
                             gridspec_kw={"hspace": 0.22})

    # Row 1 -- a single coordinate of y*(theta), to expose the staircase
    axes[0].step(thetas, Y_v[:, x_idx], where="mid",
                 color="#455A64", lw=1.2, alpha=0.9,
                 label=r"plain LP  $y^{*}_{i}(\theta)$  (vertex hops)")
    axes[0].plot(thetas, Y_cs[:, x_idx], color="#E65100", lw=1.8,
                 label=rf"cost-sketch  $\bar y_i$   (K={K}, σ={sigma:g})")
    axes[0].plot(thetas, Y_xs[:, x_idx], color="#1565C0", lw=1.8, ls="--",
                 label=rf"row-sketch   $\bar y_i$   (K={K}, k/m={k_ratio:g})")
    axes[0].set_ylabel(rf"$y^{{*}}_{{{x_idx}}}(\theta)$")
    axes[0].set_title("Coordinate of LP optimum: stochastic averaging "
                       "interpolates between vertices")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(alpha=0.3, ls=":")

    # Row 2 -- loss landscape
    axes[1].plot(thetas, L_v, color="#455A64", lw=1.2, alpha=0.9,
                 label="plain LP")
    axes[1].plot(thetas, L_cs, color="#E65100", lw=1.8,
                 label=rf"cost-sketch (K={K}, σ={sigma:g})")
    axes[1].plot(thetas, L_xs, color="#1565C0", lw=1.8, ls="--",
                 label=rf"row-sketch (K={K}, k/m={k_ratio:g})")
    axes[1].set_ylabel(r"$L(\theta) = c(\theta)^{\top}\,\bar y(\theta)$")
    axes[1].set_title("Loss landscape (still piecewise but with K-times-finer steps)")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(alpha=0.3, ls=":")

    # Row 3 -- implicit / envelope gradient g = y^T (dc/dθ)
    axes[2].plot(thetas, g_v, color="#455A64", lw=1.0, alpha=0.95,
                 label=f"plain LP   (TV={roughness(g_v):.2f})")
    axes[2].plot(thetas, g_cs, color="#E65100", lw=1.8,
                 label=f"cost-sketch (TV={roughness(g_cs):.2f})")
    axes[2].plot(thetas, g_xs, color="#1565C0", lw=1.8, ls="--",
                 label=f"row-sketch (TV={roughness(g_xs):.2f})")
    axes[2].axhline(0.0, color="k", lw=0.6, alpha=0.5)
    axes[2].set_xlabel(r"$\theta$")
    axes[2].set_ylabel(r"$g(\theta) = \bar y(\theta)^{\top}\,dc/d\theta$")
    axes[2].set_title("Implicit (autograd-style) gradient: stochastic averaging "
                       "kills the staircase jumps")
    axes[2].legend(loc="best", fontsize=9)
    axes[2].grid(alpha=0.3, ls=":")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_K_ablation(thetas, A, b, c0, c1, x_upper,
                    Ks, sigma: float, seed: int, out_path: str):
    """Roughness of the implicit gradient g(θ) vs ensemble size K."""
    _, Y_v = vanilla_sweep(thetas, A, b, c0, c1, x_upper)
    g_v = grad_implicit(thetas, Y_v, c0, c1)
    tv_v = roughness(g_v)

    tvs = []
    for K in Ks:
        _, Y_k = cost_sketch_sweep(thetas, A, b, c0, c1, x_upper,
                                   K=K, sigma=sigma, seed=seed)
        tvs.append(roughness(grad_implicit(thetas, Y_k, c0, c1)))

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.2))
    ax.axhline(tv_v, color="#455A64", ls="--", lw=1.2,
               label=f"plain LP (TV={tv_v:.2f})")
    ax.plot(Ks, tvs, "o-", color="#E65100", lw=1.8, ms=6.5,
            label=rf"cost-sketch, σ={sigma:g}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("ensemble size K")
    ax.set_ylabel(r"TV$(g(\theta))$")
    ax.set_title("Implicit-gradient roughness vs ensemble size K "
                  "(cost-side stochastic sketch)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3, ls=":")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- CLI


def main():
    parser = argparse.ArgumentParser(
        description="Toy demo: stochastic random matrices smooth the LP gradient."
    )
    parser.add_argument("--n", type=int, default=12, help="LP variables")
    parser.add_argument("--m", type=int, default=5, help="LP inequality rows")
    parser.add_argument("--lp-seed", type=int, default=3, help="Seed for (A, b)")
    parser.add_argument("--cost-seed", type=int, default=11, help="Seed for (c0, c1)")
    parser.add_argument("--n-points", type=int, default=400, help="theta grid size")
    parser.add_argument("--theta-min", type=float, default=0.0)
    parser.add_argument("--theta-max", type=float, default=2.0 * np.pi)
    parser.add_argument("--K", type=int, default=24, help="Ensemble size")
    parser.add_argument("--sigma", type=float, default=0.30,
                        help="Cost-perturbation noise scale. ||c(θ)||=1 by "
                             "construction; σ must be large enough that "
                             "perturbed costs hop across multiple vertices, "
                             "otherwise the average is just plain LP + jitter.")
    parser.add_argument("--k-ratio", type=float, default=0.8,
                        help="Row-sketch keep ratio (k/m)")
    parser.add_argument("--sketch-seed", type=int, default=0)
    parser.add_argument("--coord", type=int, default=0,
                        help="Which coordinate of y* to show in row 1")
    parser.add_argument("--out-dir", type=str,
                        default="mdp_lp/plot_results",
                        help="Where to save the figures")
    parser.add_argument("--K-ablation", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64, 128],
                        help="Ensemble sizes K used for the ablation plot")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    A, b, c0, c1, x_upper = make_lp(seed=args.lp_seed, n=args.n, m=args.m,
                                     cost_seed=args.cost_seed)
    thetas = np.linspace(args.theta_min, args.theta_max, args.n_points)

    print(f"[demo] LP shape: A={A.shape}, b={b.shape}, n={args.n}, m={args.m}")
    print(f"[demo] sweeping {args.n_points} theta values in "
          f"[{args.theta_min}, {args.theta_max}]")

    print("[demo] solving plain LP sweep...")
    L_v, Y_v = vanilla_sweep(thetas, A, b, c0, c1, x_upper)

    print(f"[demo] solving cost-sketch sweep (K={args.K}, σ={args.sigma})...")
    L_cs, Y_cs = cost_sketch_sweep(thetas, A, b, c0, c1, x_upper,
                                    K=args.K, sigma=args.sigma,
                                    seed=args.sketch_seed)

    print(f"[demo] solving row-sketch sweep (K={args.K}, k/m={args.k_ratio})...")
    L_xs, Y_xs = constraint_sketch_sweep(thetas, A, b, c0, c1, x_upper,
                                          K=args.K, k_ratio=args.k_ratio,
                                          seed=args.sketch_seed + 1)

    out_main = os.path.join(args.out_dir, "sketch_smoothing_demo.pdf")
    make_figure(thetas, L_v, L_cs, L_xs,
                K=args.K, sigma=args.sigma, k_ratio=args.k_ratio,
                x_idx=args.coord,
                Y_v=Y_v, Y_cs=Y_cs, Y_xs=Y_xs,
                c0=c0, c1=c1,
                out_path=out_main)
    print(f"[demo] wrote {out_main}")

    print("[demo] running K ablation (cost-sketch)...")
    out_ablation = os.path.join(args.out_dir, "sketch_smoothing_K_ablation.pdf")
    make_K_ablation(thetas, A, b, c0, c1, x_upper,
                    Ks=args.K_ablation, sigma=args.sigma,
                    seed=args.sketch_seed, out_path=out_ablation)
    print(f"[demo] wrote {out_ablation}")

    g_v = grad_implicit(thetas, Y_v, c0, c1)
    g_cs = grad_implicit(thetas, Y_cs, c0, c1)
    g_xs = grad_implicit(thetas, Y_xs, c0, c1)
    print()
    print("=== Roughness summary (TV of implicit gradient, lower is smoother) ===")
    print(f"  plain LP      : {roughness(g_v):8.3f}")
    print(f"  cost-sketch   : {roughness(g_cs):8.3f}  "
          f"(reduction {100*(1 - roughness(g_cs)/roughness(g_v)):+.1f}%)")
    print(f"  row-sketch    : {roughness(g_xs):8.3f}  "
          f"(reduction {100*(1 - roughness(g_xs)/roughness(g_v)):+.1f}%)")


if __name__ == "__main__":
    main()
