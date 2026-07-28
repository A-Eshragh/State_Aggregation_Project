import argparse
import random
import time
import torch
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from .configs import CartPoleConfig
from .cartpole_env import CartPoleDynamics, collect_trajectories, random_policy, heuristic_policy, epsilon_greedy_policy
from .features import ActionIndexedFeatures, ContinuousActionFeatures
from .lp_matrices import estimate_lp_matrices, estimate_lp_matrices_shared
from .reduced_solver import (
    solve_reduced_qp_gurobi, differentiable_reduced_lp, differentiable_reduced_qp,
    differentiable_unconstrained_qp, differentiable_qp_qpth,
    solve_reduced_lp_gurobi,
)
from .sketched_solver import differentiable_sketched_lp
from .policy import (
    occupancy_to_policy_fn,
    evaluate_policy,
)


def run_neural_dfl(cfg: CartPoleConfig, verbose: bool = True,
                   bins_per_dim: int = 8, tau: float = 0.5,
                   use_dfl_phase: bool = True,
                   pretrain_epochs: int = 70,
                   eval_policy_type: str = "ce",
                   behavior_policy=None,
                   n_ce_steps: int = 2,
                   n_dfl_steps: int = 5,
                   dfl_lr: float = None,
                   data: dict = None,
                   world_model=None,
                   solver: str = "lp",
                   alpha_shape: float = 2.0,
                   sa_K: int = 4,
                   sa_ratio: float = 0.2,
                   algo: str = "dfl",
                   loss_mode: str = "penalty",
                   odece_margin: float = 0.01,
                   odece_tol: float = 1e-4,
                   odece_alpha: float = 0.5,
                   odece_mse_weight: float = 0.0,
                   odece_neg_solver: str = "plain",
                   dual_weight: float = 1.0,
                   dual_feasible_only: bool = False):
    """
    algo:
        'dfl'       — differentiate through the reduced LP/QP layer (default).
        'dfl_feas'  — DFL-Feas / Odece (Mandi et al., NeurIPS'25) adapted to the
                      reduced occupancy LP: solver-free, both LP solves are
                      detached and the gradient reaches the features only
                      through the constraint residual M(θ)w − b(θ) and the
                      objective-completion MSE. See the loss block below for
                      the (measured) caveat about this setting.
    loss_mode (algo='dfl' only), mirroring mdp_lp/train_elasticity:
        'regret'      — c_true^T w*                       (no constraint term)
        'penalty'     — c_true^T w* + ρ‖b − Mw*‖²         DEFAULT, unchanged
        'lagrangian'  — c_true^T w* + ν^T(b − Mw*)
        'auglag'      — penalty + linear dual term        (proposed unbiased loss)

    Two hyperparameters damp the linear term, which on CartPole is otherwise
    both huge and signal-cancelling (measured: |linear| ~ 2-20 vs |c^T w*| ~ 5
    and rho*||viol||^2 ~ 0.003, with ~72% of true-LP solves falling back to
    Gurobi's slack-penalised model whose multipliers are bounded only by the
    1e3 slack penalty):
        dual_weight        — scalar on the linear term. 1.0 = exact augmented
                             Lagrangian (default, unchanged); 0.0 recovers
                             'penalty' exactly.
        dual_feasible_only — apply the linear term only on steps where the true
                             LP was genuinely feasible, so the multiplier is a
                             real LP dual rather than a slack-model artefact.
    """
    assert eval_policy_type in ("ce", "lp")
    assert solver in ("lp", "gurobi_qp", "qpth")
    assert algo in ("dfl", "dfl_feas")
    assert loss_mode in ("regret", "penalty", "lagrangian", "auglag")
    assert odece_neg_solver in ("plain", "arm")
    device = torch.device("cpu")
    dynamics = CartPoleDynamics(cfg)
    A = 2
    if behavior_policy is None:
        behavior_policy = heuristic_policy

    if dfl_lr is None:
        dfl_lr = cfg.lr

    if verbose:
        print("=" * 60)
        tag = "Neural DFL (CE+LP)" if use_dfl_phase else "Two-Stage (CE only)"
        print(f"Running Continuous State DFL — {tag}")
        print("=" * 60)

    # Discrete Actions
    features = ActionIndexedFeatures(
        state_dim=4, n_actions=2,
        n_features=cfg.n_features,
        hidden_dims=cfg.neural_hidden,
    ).to(device)

    # Continuous Actions
    # n_action_bases = 4
    # d_per_action = cfg.n_features // n_action_bases

    # features = ContinuousActionFeatures(
    #     state_dim=4, action_dim=1,
    #     n_action_bases=n_action_bases,
    #     d_per_action=d_per_action,
    #     hidden_dims=cfg.neural_hidden,
    #     action_low=0.0, action_high=1.0,
    #     tau_s=tau,
    # ).to(device)

    k = features.d_per_action  # state feature dimension

    if verbose:
        print("Phase 1: Pre-training features on expert actions...")

    data_provided = data is not None
    if not data_provided:
        data = collect_trajectories(dynamics, behavior_policy,
                                    cfg.n_trajectories, cfg.max_steps, device)
    buffer = {k_: v.clone() for k_, v in data.items()}

    if verbose:
        src = "externally supplied" if data_provided else "expert"
        print(f"  Loaded {buffer['states'].shape[0]} {src} transitions")

    w_head = torch.randn(k, A, device=device, requires_grad=True)
    feat_optimizer = torch.optim.Adam(
        list(features.parameters()) + [w_head], lr=cfg.lr)

    batch_size = 4096
    N = buffer["states"].shape[0]
    ce_loss = torch.nn.CrossEntropyLoss()

    for epoch in range(pretrain_epochs):
        feat_optimizer.zero_grad()

        idx = torch.randperm(N)[:batch_size]
        states_batch = buffer["states"][idx]
        expert_actions = buffer["actions"][idx].long()

        f_s = features._state_features(states_batch)  # (B, k)
        logits = f_s @ w_head                          # (B, A)
        loss = ce_loss(logits, expert_actions)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(features.parameters(), max_norm=1.0)
        feat_optimizer.step()

        if verbose and (epoch + 1) % 50 == 0:
            with torch.no_grad():
                acc = (logits.argmax(1) == expert_actions).float().mean()
            print(f"  [Pretrain] Epoch {epoch+1}/{pretrain_epochs}, "
                  f"loss={loss.item():.4f}, acc={acc.item():.3f}")

    if verbose:
        phase_desc = ("Phase 2: DFL iterations (LP + constraint penalty)..."
                      if use_dfl_phase else
                      "Phase 2: CE-only (two-stage) iterations...")
        print(phase_desc)

    if not data_provided:
        warmup = collect_trajectories(dynamics, behavior_policy,
                                      5, cfg.max_steps, device)
        for key in ["states", "actions", "rewards", "next_states", "dones"]:
            buffer[key] = torch.cat([buffer[key], warmup[key]], dim=0)
        buffer["init_states"] = torch.cat(
            [buffer["init_states"], warmup["init_states"]], dim=0)

    if world_model is not None:
        with torch.no_grad():
            delta_pred, r_pred = world_model.predict(
                buffer["states"], buffer["actions"].long())
            buffer["next_states"] = buffer["states"] + delta_pred
            buffer["rewards"] = r_pred
            buffer["dones"] = dynamics.is_terminal(buffer["next_states"])
        if verbose:
            print(f"  [world-model] Replaced (r, s') in {buffer['states'].shape[0]} "
                  f"transitions with deterministic model predictions")

    N = buffer["states"].shape[0]
    if verbose:
        print(f"  Offline dataset frozen: {N} transitions, "
              f"{buffer['init_states'].shape[0]} initial states")

    # dfl_optimizer = torch.optim.Adam(features.parameters(), lr=cfg.lr)
    dfl_optimizer = torch.optim.Adam(list(features.parameters()) + [w_head], lr=dfl_lr)
    rho = cfg.rho
    lp_weight = 1

    if solver == "gurobi_qp":
        # Reduced QP solved by Gurobi's interior-point: min c^T w + (eta/2)||w||^2
        # s.t. Mw=b, w>=0. Custom KKT-based backward in DifferentiableReducedQP.
        qp_eta = float(cfg.qp_eta)
        def solve_lp(M, c, b):
            return differentiable_reduced_qp(M, c, b, eta=qp_eta)
    elif solver == "qpth":
        # Real qpth library — autograd-native PDIPM, no Gurobi.
        qp_eta = float(cfg.qp_eta)
        def solve_lp(M, c, b):
            w_star, _ = differentiable_qp_qpth(M, c, b, eta=qp_eta)
            return w_star
    else:
        # solver == "lp": sketch-and-solve LP — K independent row-sketched
        # reduced LPs, ensemble-averaged. Smooths over basis-selection
        # degeneracy that the dense single-LP path was sensitive to.
        _K, _r = int(sa_K), float(sa_ratio)
        def solve_lp(M, c, b):
            return differentiable_sketched_lp(
                M, c, b,
                n_sketches=_K, sketch_ratio=_r, sketch_mode="gaussian")

    def solve_eval(M, c, b):
        return differentiable_reduced_lp(M, c, b)

    def compute_shaped_rewards(states, actions, rewards):
        with torch.no_grad():
            f_s = features._state_features(states)
            logits = f_s @ w_head
            probs = torch.softmax(logits, dim=1)
            bonus = probs.gather(1, actions.long().unsqueeze(1)).squeeze(1)
        return rewards + alpha_shape * bonus

    with torch.no_grad():
        shaped_buffer = {kk: v.clone() for kk, v in buffer.items()}
        shaped_buffer["rewards"] = compute_shaped_rewards(
            buffer["states"], buffer["actions"], buffer["rewards"])
        M, c, b = estimate_lp_matrices(
            features, shaped_buffer, gamma=cfg.gamma, ridge=cfg.ridge)
        w_star = solve_eval(M, c, b).detach()
    current_policy = occupancy_to_policy_fn(features, w_star)

    def ce_policy(states):
        with torch.no_grad():
            f_s = features._state_features(states)
            return (f_s @ w_head).argmax(dim=1)

    eval_policy = ce_policy if eval_policy_type == "ce" else current_policy

    if verbose:
        test_s = torch.tensor([
            [0.0, 0.0, 0.05, 0.0],
            [0.0, 0.0, -0.05, 0.0],
        ], device=device)
        acts = current_policy(test_s)
        expert_acts = heuristic_policy(test_s)
        ws_results = evaluate_policy(
            dynamics, eval_policy, 50, cfg.max_steps, cfg.gamma, device)
        print(f"  Warm-start: acts={acts.tolist()} (expert={expert_acts.tolist()}), "
              f"reward={ws_results['avg_reward']:.1f}")

    def snapshot():
        return (
            {kk: v.detach().clone() for kk, v in features.state_dict().items()},
            w_star.detach().clone(),
            w_head.detach().clone(),
        )

    best_reward = ws_results["avg_reward"] if verbose else -float("inf")
    best_state, best_w, best_head = snapshot()

    solver_forward_time = 0.0
    backward_time = 0.0
    # One-shot strong-duality gate for the 'lagrangian'/'auglag' linear term
    # (list so the inner loop can flip it).
    _dual_gate_done = [False]
    # (# true-LP solves that hit Gurobi's slack fallback, # true-LP solves)
    _dual_stats = [0, 0]
    # DFL-Feas gradient audit: fraction of DFL steps whose Odece loss produced a
    # non-negligible feature gradient. Records, rather than assumes, whether the
    # baseline actually trains in this setting.
    _feas_grad_hits = [0, 0]

    history = []
    for iteration in range(cfg.n_iterations):
        t0 = time.time()

        if eval_policy_type == "lp":
            eval_policy = occupancy_to_policy_fn(features, w_star)

        results = evaluate_policy(
            dynamics, eval_policy, cfg.n_eval_episodes,
            cfg.max_steps, cfg.gamma, device)
        if results["avg_reward"] > best_reward:
            best_reward = results["avg_reward"]
            best_state, best_w, best_head = snapshot()

        with torch.no_grad():
            dagger_actions = current_policy(buffer["states"]).long()
        dagger_states = buffer["states"]

        # Phase A: CE training
        ce_log = 0.0
        for _ in range(n_ce_steps):
            dfl_optimizer.zero_grad()
            ce_idx = torch.randperm(dagger_states.shape[0])[:batch_size]
            f_s = features._state_features(dagger_states[ce_idx])
            logits = f_s @ w_head
            ce_aux = ce_loss(logits, dagger_actions[ce_idx])
            ce_aux.backward()
            torch.nn.utils.clip_grad_norm_(features.parameters(), max_norm=1.0)
            dfl_optimizer.step()
            ce_log = ce_aux.item()

        # Phase B: DFL
        lp_obj_log = 0.0
        aug_lag_log = 0.0
        lin_log = 0.0
        feas_diag = None
        if use_dfl_phase:
            for _ in range(n_dfl_steps):
                dfl_optimizer.zero_grad()

                if N > batch_size:
                    idx = torch.randperm(N)[:batch_size]
                    batch = {
                        "states": buffer["states"][idx],
                        "actions": buffer["actions"][idx],
                        "rewards": buffer["rewards"][idx],
                        "next_states": buffer["next_states"][idx],
                        "dones": buffer["dones"][idx],
                        "init_states": buffer["init_states"],
                    }
                else:
                    batch = buffer

                shaped_rewards = compute_shaped_rewards(
                    batch["states"], batch["actions"], batch["rewards"])

                # M and b don't depend on rewards — estimate once, derive
                # c_shaped (for the LP forward) and c_true (for the loss)
                # from the same Phi_sa with two cheap reductions.
                M, b, Phi_sa = estimate_lp_matrices_shared(
                    features, batch, gamma=cfg.gamma, ridge=cfg.ridge)
                N_b = Phi_sa.shape[0]
                c_shaped = -(Phi_sa * shaped_rewards.unsqueeze(1)).sum(dim=0) / N_b
                c_true = -(Phi_sa * batch["rewards"].unsqueeze(1)).sum(dim=0) / N_b

                if algo == "dfl_feas":
                    # ---- DFL-Feas / Odece, adapted to the reduced occupancy LP ----
                    # Solver-free: BOTH solves are detached, so no gradient ever
                    # flows through the LP layer. The features are reached only via
                    #   c(w; θ) = M(θ) w − b(θ)
                    # and the objective-completion MSE ‖c_shaped − c_true‖²
                    # (the analogue of mdp_lp's μ‖r_pred − r_true‖²: c_shaped is the
                    # predicted objective the layer optimises, c_true the real one).
                    #
                    # y^- (negative example) = the solution of the PREDICTED
                    # problem (M, c_shaped, b). odece_neg_solver='plain' solves it
                    # exactly, which is what [34] specifies; 'arm' instead reuses
                    # this arm's (sketched) layer, a charitable variant that gives
                    # y^- a non-zero residual for IPL to act on.
                    # y_true (positive example) = the plain reduced LP under the
                    # true objective.
                    _t = time.time()
                    with torch.no_grad():
                        if odece_neg_solver == "arm":
                            w_neg = solve_lp(M, c_shaped, b)
                        else:
                            w_neg, _, _ = solve_reduced_lp_gurobi(M, c_shaped, b)
                        w_pos, _, _ = solve_reduced_lp_gurobi(M, c_true, b)
                    solver_forward_time += time.time() - _t

                    ups, tol = float(odece_margin), float(odece_tol)
                    # Adaptation caveat, worth stating plainly: in mdp_lp the two
                    # Odece losses are driven by the gap between a PREDICTED and a
                    # TRUE constraint system (H_pred vs H_true). Here there is only
                    # one, sample-estimated system (M, b), shared by both problems.
                    # What keeps the losses alive is that the flow polytope
                    # {w : Mw = b, w >= 0} is typically EMPTY once the features are
                    # pretrained, so solve_reduced_lp_gurobi falls back to its
                    # slack-penalised model and returns points with a genuinely
                    # non-zero residual. Measured on the paper's config (nfeat=1024,
                    # seed 1): the fallback fires on ~87-100% of solves and 100% of
                    # DFL steps produce a non-zero feature gradient. So both terms
                    # act on real residuals — but they push the residual toward
                    # zero, i.e. they carry feasibility signal only, with no
                    # decision-value term anywhere in the loss.
                    #
                    # OPL: the true optimum should satisfy the flow constraints.
                    c_pos = M @ w_pos - b
                    opl = (torch.nn.functional.softplus(ups + c_pos)
                           + torch.nn.functional.softplus(ups - c_pos)).sum()
                    # IPL: sign-selected single-sided softplus. The UNSAT side is
                    # picked from the residual of y^- on the true constraint
                    # system, which here is c_neg.detach(). Its one-sided
                    # derivative is non-zero even at c = 0 (it is sigma(ups)), so
                    # IPL keeps pushing y^-'s residual toward zero — a solver-free
                    # surrogate for the rho||.||^2 penalty.
                    c_neg = M @ w_neg - b
                    v_neg = c_neg.detach()
                    unsat_pos = (v_neg > tol).float()
                    unsat_neg = (v_neg < -tol).float()
                    infeas = (v_neg.abs().max() > tol).float()
                    ipl = (infeas * (unsat_pos * torch.nn.functional.softplus(ups - c_neg)
                                     + unsat_neg * torch.nn.functional.softplus(ups + c_neg))).sum()
                    loss = float(odece_alpha) * ipl + (1.0 - float(odece_alpha)) * opl
                    if odece_mse_weight > 0:
                        loss = loss + float(odece_mse_weight) * \
                            torch.nn.functional.mse_loss(c_shaped, c_true)
                    loss = loss * lp_weight

                    with torch.no_grad():
                        lp_obj = c_true @ w_neg
                        aug_lag = rho * (b - M @ w_neg).pow(2).sum()
                    feas_diag = (float(opl.item()), float(ipl.item()),
                                 float(c_pos.detach().norm().item()),
                                 float(c_neg.detach().norm().item()))
                else:
                    _t = time.time()
                    w_star_inner = solve_lp(M, c_shaped, b)
                    solver_forward_time += time.time() - _t

                    lp_obj = c_true @ w_star_inner
                    violation = b - M @ w_star_inner
                    aug_lag = rho * violation.pow(2).sum()

                    # loss_mode ablation. 'penalty' reproduces the original
                    # Eq.(5) loss EXACTLY, so runs that do not set --loss_mode
                    # are byte-for-byte unchanged.
                    loss = lp_obj
                    if loss_mode in ("penalty", "auglag"):
                        loss = loss + aug_lag
                    if loss_mode in ("lagrangian", "auglag"):
                        # Linear dual term ν^T(b − M w*) = ν^T·violation, where ν
                        # is the dual of the TRUE reduced LP (M, c_true, b),
                        # re-solved fresh under no_grad each step (M, b move with
                        # θ, so it is a constant wrt θ only within the step).
                        # Sign note: this is +ν·violation, not mdp_lp's
                        # −v*·violation, because the reduced LP is stated as a
                        # MINIMISATION (c = −reward) whereas the occupancy LP in
                        # mdp_lp is a maximisation; the two Lagrangians agree
                        # after ν ↦ −v*.
                        with torch.no_grad():
                            w_true_lp, _, nu_true = solve_reduced_lp_gurobi(
                                M.detach(), c_true.detach(), b.detach())
                        # Track how often the exact LP was infeasible and Gurobi
                        # fell back to its slack-penalised model: that model's
                        # multipliers are only bounded by the slack penalty, so
                        # the linear term can get large. Recorded, not clipped.
                        _res = float((b.detach() - M.detach() @ w_true_lp)
                                     .norm().item())
                        _dual_stats[1] += 1
                        _fell_back = _res > 1e-6 * max(
                            float(b.detach().norm()), 1e-12)
                        if _fell_back:
                            _dual_stats[0] += 1

                        if not _dual_gate_done[0]:
                            # KKT gate on the multiplier's sign/convention. We
                            # check DUAL FEASIBILITY + COMPLEMENTARITY, not
                            # strong duality: when the exact reduced LP is
                            # infeasible, solve_reduced_lp_gurobi falls back to a
                            # slack-penalised model whose optimal value absorbs
                            # the slack penalty, so b^T nu != c^T w* there even
                            # though nu is still a valid KKT multiplier. Dual
                            # feasibility is the sign-discriminative property:
                            # it is what makes (nu * violation) the standard
                            # augmented-Lagrangian term nu_std^T(Mw - b) with
                            # nu_std = -nu, i.e. bias-cancelling rather than
                            # bias-doubling. Flipping the sign breaks it at once.
                            red = c_true.detach() - M.detach().T @ nu_true
                            scale = max(float(c_true.detach().abs().max()), 1e-12)
                            dual_feas = float(red.min().item())
                            comp = float((red * w_true_lp).abs().sum().item())
                            obj = abs(float((c_true.detach() * w_true_lp).sum().item()))
                            if dual_feas < -1e-4 * scale - 1e-9:
                                raise RuntimeError(
                                    "Linear-term dual-feasibility check failed: "
                                    f"min(c_true - M^T nu) = {dual_feas:.3e} "
                                    f"(scale {scale:.3e}); the dual's sign "
                                    "convention is inconsistent with (M, b) — "
                                    "auglag results untrustworthy.")
                            if comp > 1e-3 * max(obj, 1e-12) + 1e-6:
                                raise RuntimeError(
                                    "Linear-term complementarity check failed: "
                                    f"|(c_true - M^T nu) . w*| = {comp:.3e} vs "
                                    f"|c^T w*| = {obj:.3e}; nu is not a valid KKT "
                                    "multiplier — auglag results untrustworthy.")
                            _dual_gate_done[0] = True
                        # dual_weight damps the term; dual_feasible_only drops it
                        # on steps whose multiplier came from Gurobi's slack
                        # model rather than a genuine LP dual. Both default to
                        # the exact augmented Lagrangian (1.0 / False).
                        _dw = float(dual_weight)
                        if dual_feasible_only and _fell_back:
                            _dw = 0.0
                        linear = _dw * (nu_true * violation).sum()
                        lin_log = float(linear.item())
                        loss = loss + linear
                    loss = loss * lp_weight
                    feas_diag = None
                _t = time.time()
                loss.backward()
                backward_time += time.time() - _t
                gnorm = torch.nn.utils.clip_grad_norm_(
                    features.parameters(), max_norm=1.0)
                dfl_optimizer.step()

                if algo == "dfl_feas":
                    _feas_grad_hits[1] += 1
                    if float(gnorm) > 1e-8:
                        _feas_grad_hits[0] += 1
                    grad_norm_log = float(gnorm)

                lp_obj_log = lp_obj.item()
                aug_lag_log = aug_lag.item()

        with torch.no_grad():
            shaped_full = {kk: v for kk, v in buffer.items()}
            shaped_full["rewards"] = compute_shaped_rewards(
                buffer["states"], buffer["actions"], buffer["rewards"])
            M_eval, c_eval, b_eval = estimate_lp_matrices(
                features, shaped_full, gamma=cfg.gamma, ridge=cfg.ridge)
            w_star = solve_eval(M_eval, c_eval, b_eval).detach()
        current_policy = occupancy_to_policy_fn(features, w_star)

        elapsed = time.time() - t0
        history.append({"iteration": iteration, **results})

        if verbose:
            w_nnz = int((w_star > 1e-8).sum())
            extra = ""
            if loss_mode in ("lagrangian", "auglag"):
                extra = f" | linear={lin_log:+.4f}"
            if feas_diag is not None:
                extra = (f" | opl={feas_diag[0]:.4f} ipl={feas_diag[1]:.4f}"
                         f" |res_pos|={feas_diag[2]:.2e} |res_neg|={feas_diag[3]:.2e}"
                         f" gnorm={grad_norm_log:.2e}")
            print(
                f"  Iter {iteration:3d} | reward={results['avg_reward']:.2f} | "
                f"length={results['avg_length']:.1f} | "
                f"lp_obj={lp_obj_log:.4f} | aug_lag={aug_lag_log:.4f} | "
                f"ce={ce_log:.4f} | nnz={w_nnz}/{features.n_features} | "
                f"time={elapsed:.1f}s{extra}"
            )

    # Final eval
    if eval_policy_type == "lp":
        eval_policy = occupancy_to_policy_fn(features, w_star)
    final_results = evaluate_policy(
        dynamics, eval_policy, cfg.n_eval_episodes,
        cfg.max_steps, cfg.gamma, device)
    if final_results["avg_reward"] > best_reward:
        best_reward = final_results["avg_reward"]
    else:
        features.load_state_dict(best_state)
        w_star = best_w
        with torch.no_grad():
            w_head.copy_(best_head)
        current_policy = occupancy_to_policy_fn(features, w_star)
    if verbose:
        print(f"  Best reward over run: {best_reward:.2f}")

    return {"history": history, "features": features, "w_star": w_star,
            "w_head": w_head, "best_reward": best_reward,
            "solver_forward_time": float(solver_forward_time),
            "backward_time": float(backward_time),
            "solver_total_time": float(solver_forward_time + backward_time),
            # DFL-Feas only: (# steps with non-zero feature gradient, # steps)
            "feas_grad_hits": int(_feas_grad_hits[0]),
            "feas_grad_steps": int(_feas_grad_hits[1]),
            # auglag/lagrangian only: how often the true-LP dual solve hit
            # Gurobi's slack-penalised fallback (multiplier then bounded only
            # by the slack penalty).
            "dual_fallbacks": int(_dual_stats[0]),
            "dual_solves": int(_dual_stats[1])}


def run_two_stage_mbrl(cfg: CartPoleConfig, verbose: bool = True,
                       bins_per_dim: int = 8, tau: float = 0.5,
                       pretrain_epochs: int = 0,
                       eval_policy_type: str = "lp",
                       behavior_policy=None,
                       n_ce_steps: int = 0,
                       n_dfl_steps: int = 0,
                       dfl_lr: float = None,
                       data: dict = None,
                       world_model=None,
                       world_epochs: int = 400,
                       world_lr: float = 1e-3,
                       q_hidden=(128, 128),
                       q_lr: float = 3e-4,
                       q_batch_size: int = 256,
                       n_q_steps_per_iter: int = 128,
                       rollouts_per_iter: int = 0,
                       rollout_length: int = 1,
                       n_initial_synthetic: int = 0,
                       syn_buffer_max: int = 200_000,
                       eps_greedy: float = 0.1,
                       target_update_iters: int = 1,
                       bc_alpha: float = 0.0,
                       bc_pretrain_epochs: int = 30):
    import torch.nn as nn
    import torch.nn.functional as F
    from .world_model import train_world_model

    del pretrain_epochs, eval_policy_type, n_ce_steps, n_dfl_steps, dfl_lr  # unused
    del bins_per_dim, tau

    device = torch.device("cpu")
    dynamics = CartPoleDynamics(cfg)
    A = 2
    if behavior_policy is None:
        behavior_policy = heuristic_policy

    if verbose:
        print("=" * 60)
        print("Two-Stage MBRL — DQN on a frozen learned world model")
        print("=" * 60)

    # ── 1. Offline buffer ───────────────────────────────────────────
    if data is None:
        if verbose:
            print(f"  Collecting {cfg.n_trajectories} offline trajectories...")
        data = collect_trajectories(dynamics, behavior_policy,
                                    cfg.n_trajectories, cfg.max_steps, device)
    real = {k_: v.clone() for k_, v in data.items()}
    state_dim = real["states"].shape[1]
    state_scales = torch.tensor([2.4, 3.0, 0.2095, 3.0], device=device)

    if verbose:
        print(f"  Real buffer: {real['states'].shape[0]} transitions, "
              f"{real['init_states'].shape[0]} initial states")

    # ── 2. World model (fit then freeze) ────────────────────────────
    if world_model is None:
        if verbose:
            print(f"  Fitting world model ({world_epochs} epochs)...")
        world_model = train_world_model(real, n_epochs=world_epochs,
                                        lr=world_lr, verbose=verbose)
    for p in world_model.parameters():
        p.requires_grad_(False)
    world_model.eval()

    # ── 3. Q-network + target network ───────────────────────────────
    def make_q_net():
        layers, prev = [], state_dim
        for h in q_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, A))
        return nn.Sequential(*layers).to(device)

    q_net = make_q_net()
    q_target = make_q_net()
    q_target.load_state_dict(q_net.state_dict())
    for p in q_target.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=q_lr)

    def q_policy(states, eps):
        with torch.no_grad():
            qvals = q_net(states / state_scales)
            greedy = qvals.argmax(dim=1)
            if eps <= 0.0:
                return greedy
            rand = torch.randint(0, A, (states.shape[0],), device=device)
            mask = torch.rand(states.shape[0], device=device) < eps
            return torch.where(mask, rand, greedy)

    # ── 4. Replay (pre-allocated, FIFO) ─────────────────────────────
    # Seed the replay with the real offline buffer. Without this, the Q
    # net only sees rollouts under ε-greedy(Q) where Q is random at
    # init — those rollouts are near-uniform, and the Q just learns the
    # value of a random policy (~9 steps on CartPole). Mixing in the
    # logged behavior-policy transitions is the standard offline-MBRL
    # recipe (cf. MBPO / MOPO use real + model data).
    rb_s  = torch.zeros(syn_buffer_max, state_dim, device=device)
    rb_a  = torch.zeros(syn_buffer_max, dtype=torch.long, device=device)
    rb_r  = torch.zeros(syn_buffer_max, device=device)
    rb_sn = torch.zeros(syn_buffer_max, state_dim, device=device)
    rb_d  = torch.zeros(syn_buffer_max, dtype=torch.bool, device=device)
    rb_pos, rb_size = 0, 0

    def rb_add(s, a, r, sn, d):
        nonlocal rb_pos, rb_size
        n = s.shape[0]
        if n == 0:
            return
        # Wrap around in up to two contiguous slices.
        idx = (torch.arange(n, device=device) + rb_pos) % syn_buffer_max
        rb_s[idx]  = s
        rb_a[idx]  = a
        rb_r[idx]  = r
        rb_sn[idx] = sn
        rb_d[idx]  = d
        rb_pos = int((rb_pos + n) % syn_buffer_max)
        rb_size = int(min(rb_size + n, syn_buffer_max))

    # Optional BC reward shaping: train a BC head on the offline data
    # (same recipe as run_neural_dfl's pretrain phase) and add
    # bc_alpha * pi_bc(a|s) to every transition's reward. This lets MBRL
    # receive the same imitation signal that DFL gets from
    # compute_shaped_rewards, so the Two-stage vs DFL comparison is not
    # confounded by an asymmetric BC bonus.
    bc_features = None
    bc_w_head = None
    if bc_alpha > 0.0:
        bc_features = ActionIndexedFeatures(
            state_dim=state_dim, n_actions=A,
            n_features=cfg.n_features,
            hidden_dims=cfg.neural_hidden,
        ).to(device)
        k_bc = bc_features.d_per_action
        bc_w_head = torch.randn(k_bc, A, device=device, requires_grad=True)
        bc_opt = torch.optim.Adam(
            list(bc_features.parameters()) + [bc_w_head], lr=cfg.lr)
        bc_ce = nn.CrossEntropyLoss()
        N_bc = real["states"].shape[0]
        bc_batch = min(4096, N_bc)
        for _ in range(bc_pretrain_epochs):
            bc_opt.zero_grad()
            idx = torch.randperm(N_bc)[:bc_batch]
            f_s = bc_features._state_features(real["states"][idx])
            logits = f_s @ bc_w_head
            loss = bc_ce(logits, real["actions"][idx].long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bc_features.parameters(), max_norm=1.0)
            bc_opt.step()
        for p in bc_features.parameters():
            p.requires_grad_(False)
        bc_w_head.requires_grad_(False)
        if verbose:
            with torch.no_grad():
                acc = (logits.argmax(1) == real["actions"][idx].long()
                       ).float().mean().item()
            print(f"  [BC shaping] pretrained head, acc={acc:.3f}, "
                  f"alpha={bc_alpha}")

    def _bc_bonus(states, actions):
        if bc_features is None:
            return torch.zeros(states.shape[0], device=device)
        with torch.no_grad():
            probs = torch.softmax(
                bc_features._state_features(states) @ bc_w_head, dim=1)
            return bc_alpha * probs.gather(
                1, actions.long().unsqueeze(1)).squeeze(1)

    real_shaped_rewards = real["rewards"] + _bc_bonus(
        real["states"], real["actions"].long())
    rb_add(real["states"], real["actions"].long(), real_shaped_rewards,
           real["next_states"], real["dones"].bool())
    if verbose:
        print(f"  Replay seeded with {rb_size} real transitions"
              f"{' (shaped)' if bc_alpha > 0 else ''}")

    # ── 4b. One-shot synthetic data via the frozen world model ─────
    # Roll out the behavior policy in the model to expand state-action
    # coverage without distribution shift. This is the model-based step:
    # the Q-net later trains on real + model-augmented data via FQI.
    if n_initial_synthetic > 0:
        from .world_model import generate_synthetic_buffer
        n_branches = max(n_initial_synthetic // max(rollout_length, 1), 1)
        with torch.no_grad():
            syn = generate_synthetic_buffer(
                world_model, real, n_total=n_initial_synthetic,
                terminal_fn=dynamics.is_terminal,
                rollout_length=max(rollout_length, 1),
                action_policy=lambda s: behavior_policy(s).long())
        if syn["states"].numel() > 0:
            syn_shaped_rewards = syn["rewards"] + _bc_bonus(
                syn["states"], syn["actions"].long())
            rb_add(syn["states"], syn["actions"].long(), syn_shaped_rewards,
                   syn["next_states"], syn["dones"].bool())
            if verbose:
                print(f"  Replay augmented with {syn['states'].shape[0]} "
                      f"synthetic transitions (rollout_length="
                      f"{max(rollout_length, 1)})")

    def rb_sample(B):
        idx = torch.randint(0, rb_size, (B,), device=device)
        return rb_s[idx], rb_a[idx], rb_r[idx], rb_sn[idx], rb_d[idx]

    # ── 5. Dyna outer loop ──────────────────────────────────────────
    history = []
    best_reward = -float("inf")
    real_states = real["states"]
    N_real = real_states.shape[0]
    init_states_real = real["init_states"]

    for iteration in range(cfg.n_iterations):
        t0 = time.time()

        # Rollout k steps in the model under the behavior policy. Using
        # the offline behavior policy (rather than ε-greedy(Q)) keeps the
        # synthetic data in-distribution; ε-greedy on a random-init Q-net
        # generates near-uniform actions that drag Q to the value of a
        # random policy. This is the simplest stable MBRL recipe.
        if rollouts_per_iter > 0 and rollout_length > 0:
            starts = torch.randint(0, N_real, (rollouts_per_iter,),
                                   device=device)
            s = real_states[starts].clone()
            alive = torch.ones(rollouts_per_iter, dtype=torch.bool,
                               device=device)
            with torch.no_grad():
                for _ in range(rollout_length):
                    if not alive.any():
                        break
                    s_act = s[alive]
                    a_act = behavior_policy(s_act).long()
                    sn_act, r_act = world_model.sample_next(
                        s_act, a_act, deterministic=True)
                    done_act = dynamics.is_terminal(sn_act)
                    r_shaped = r_act + _bc_bonus(s_act, a_act)
                    rb_add(s_act, a_act, r_shaped, sn_act, done_act)
                    s_new = s.clone()
                    s_new[alive] = sn_act
                    s = s_new
                    alive_next = alive.clone()
                    alive_next[alive] = ~done_act
                    alive = alive_next

        # DQN updates against the target net.
        loss_log = 0.0
        if rb_size >= q_batch_size:
            for _ in range(n_q_steps_per_iter):
                bs, ba, br, bsn, bd = rb_sample(q_batch_size)
                with torch.no_grad():
                    q_next = q_target(bsn / state_scales).max(dim=1).values
                    target = br + cfg.gamma * (~bd).float() * q_next
                q_pred = q_net(bs / state_scales).gather(
                    1, ba.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(q_pred, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
                optimizer.step()
                loss_log = loss.item()

        if (iteration + 1) % target_update_iters == 0:
            q_target.load_state_dict(q_net.state_dict())

        # Evaluate greedy(Q) on the *real* env.
        results = evaluate_policy(
            dynamics, lambda st: q_policy(st, eps=0.0),
            cfg.n_eval_episodes, cfg.max_steps, cfg.gamma, device)
        if results["avg_reward"] > best_reward:
            best_reward = results["avg_reward"]
        history.append({"iteration": iteration, **results})

        if verbose:
            elapsed = time.time() - t0
            print(f"  Iter {iteration:3d} | reward={results['avg_reward']:.2f} | "
                  f"length={results['avg_length']:.1f} | "
                  f"q_loss={loss_log:.4f} | "
                  f"syn={rb_size:>6d} | time={elapsed:.1f}s")

    return {"history": history,
            "q_net": q_net,
            "world_model": world_model,
            "best_reward": float(best_reward),
            "solver_total_time": 0.0}


def plot_learning_curves(history: list, save_path: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    xs = list(range(len(history)))
    rewards = [h["avg_reward"] for h in history]
    lengths = [h["avg_length"] for h in history]
    ax1.plot(xs, rewards, marker="o", markersize=3)
    ax2.plot(xs, lengths, marker="o", markersize=3)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Average Episode Reward")
    ax1.set_title("Learning Curve — Reward")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Average Episode Length")
    ax2.set_title("Learning Curve — Episode Length")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    plt.show()


def parse_args():
    p = argparse.ArgumentParser(description="Continuous MDP DFL on CartPole")
    p.add_argument("--method", type=str, default="soft_agg",
                   choices=["neural_dfl", "soft_agg", "two_stage_mbrl"],
                   help="Training method")
    # Shared
    p.add_argument("--n_iterations", type=int, default=100)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--qp_eta", type=float, default=0.01)
    p.add_argument("--reward_type", type=str, default="cos_theta",
                   choices=["cos_theta", "constant"])
    p.add_argument("--save_plot", type=str, default=None)
    p.add_argument("--n_eval_episodes", type=int, default=50)
    # Neural DFL
    p.add_argument("--n_features", type=int, default=128)
    p.add_argument("--n_trajectories", type=int, default=500)
    p.add_argument("--n_epochs", type=int, default=100)
    p.add_argument("--n_dfl_steps", type=int, default=5)
    p.add_argument("--n_ce_steps", type=int, default=0)
    p.add_argument("--ridge", type=float, default=1e-4)
    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--pretrain_epochs", type=int, default=70)
    p.add_argument("--eval_policy", type=str, default="lp", choices=["ce", "lp"])
    # Soft aggregation
    p.add_argument("--n_clusters", type=int, default=32)
    p.add_argument("--bins_per_dim", type=int, default=8)
    p.add_argument("--tau", type=float, default=0.5,
                   help="Softmax temperature for membership")
    p.add_argument("--seed", type=int, default=1,
                   help="Random seed for reproducibility")
    # World-model knobs — every method that consumes a learned MDP model
    # uses one unconditionally (neural_dfl swaps the LP estimator's
    # (r, s') for model predictions; two_stage_mbrl runs DQN on the
    # frozen model). soft_agg uses the discretised true dynamics and so
    # has no learned model.
    p.add_argument("--world_epochs", type=int, default=400,
                   help="World-model training epochs (Gaussian dynamics + "
                        "MSE reward, val early-stop).")
    p.add_argument("--world_lr", type=float, default=1e-3,
                   help="World-model Adam learning rate.")
    p.add_argument("--mbrl_q_hidden", type=int, nargs="+", default=[32, 32],
                   help="MBRL Q-net hidden layer sizes.")
    p.add_argument("--mbrl_q_lr", type=float, default=3e-4)
    p.add_argument("--mbrl_q_steps", type=int, default=128,
                   help="DQN gradient steps per outer iteration.")
    p.add_argument("--mbrl_q_batch_size", type=int, default=256)
    p.add_argument("--mbrl_rollouts_per_iter", type=int, default=0,
                   help="Per-iter k-step model rollouts. Off by default — "
                        "see run_two_stage_mbrl docstring for why.")
    p.add_argument("--mbrl_rollout_length", type=int, default=1)
    p.add_argument("--mbrl_n_initial_synthetic", type=int, default=0,
                   help="One-shot synthetic transitions to add to the "
                        "replay before training. Off by default.")
    p.add_argument("--mbrl_eps_greedy", type=float, default=0.1)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = CartPoleConfig(
        n_iterations=args.n_iterations,
        gamma=args.gamma,
        lr=args.lr,
        qp_eta=args.qp_eta,
        reward_type=args.reward_type,
        n_features=args.n_features,
        n_trajectories=args.n_trajectories,
        n_epochs=args.n_epochs,
        ridge=args.ridge,
        rho=10.0,
        n_eval_episodes=args.n_eval_episodes,
    )

    if args.method == "two_stage_mbrl":
        out = run_two_stage_mbrl(
            cfg, verbose=True,
            bins_per_dim=args.bins_per_dim, tau=args.tau,
            world_epochs=args.world_epochs,
            world_lr=args.world_lr,
            q_hidden=tuple(args.mbrl_q_hidden),
            q_lr=args.mbrl_q_lr,
            q_batch_size=args.mbrl_q_batch_size,
            n_q_steps_per_iter=args.mbrl_q_steps,
            rollouts_per_iter=args.mbrl_rollouts_per_iter,
            rollout_length=args.mbrl_rollout_length,
            n_initial_synthetic=args.mbrl_n_initial_synthetic,
            eps_greedy=args.mbrl_eps_greedy,
        )
    else:
        from .world_model import train_world_model
        device = torch.device("cpu")
        dynamics = CartPoleDynamics(cfg)
        buf = collect_trajectories(dynamics, heuristic_policy,
                                   cfg.n_trajectories, cfg.max_steps,
                                   device)
        print(f"[WM] Fitting world model on {buf['states'].shape[0]} "
              f"transitions for {args.world_epochs} epochs...")
        wm = train_world_model(buf, n_epochs=args.world_epochs,
                               lr=args.world_lr, verbose=True)

        out = run_neural_dfl(
            cfg, verbose=True,
            bins_per_dim=args.bins_per_dim, tau=args.tau,
            pretrain_epochs=args.pretrain_epochs,
            eval_policy_type=args.eval_policy,
            n_ce_steps=args.n_ce_steps,
            n_dfl_steps=args.n_dfl_steps,
            data=buf,
            world_model=wm,
        )

    plot_learning_curves(out["history"], save_path=args.save_plot)


if __name__ == "__main__":
    main()