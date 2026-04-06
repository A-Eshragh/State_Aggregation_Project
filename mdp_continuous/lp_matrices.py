"""
Sample-based estimation of the reduced LP system: Mw = b, min c^T w.

From the projected flow constraint (continuous_mdp.tex):
    M = Psi^T (B - gamma P) Phi   in R^{m x d}
    c_j = -integral r(s,a) phi_j(s,a) ds da  (negative for minimization)
    b_i = integral psi_i(s) rho(s) ds

Estimated from trajectory data {(s_t, a_t, r_t, s_{t+1}, done_t)}.
"""

import torch


def estimate_lp_matrices(
    feature_model,
    trajectory_data: dict,
    gamma: float = 0.99,
    n_actions: int = 2,
    ridge: float = 1e-4,
):
    """Estimate M, c, b from trajectory data.

    Args:
        feature_model: object with .phi(states, actions) and .psi(states)
        trajectory_data: dict with keys:
            states:      (N, 4)
            actions:     (N,)
            rewards:     (N,)
            next_states: (N, 4)
            dones:       (N,) bool
            init_states: (N_0, 4)
        gamma: discount factor
        n_actions: number of actions
        ridge: small diagonal regularization added to M for numerical stability

    Returns:
        M: (m, d) constraint matrix
        c: (d,)   cost vector (for minimization: c = -reward coefficients)
        b: (m,)   RHS vector (from initial distribution)
    """
    states = trajectory_data["states"]           # (N, 4)
    actions = trajectory_data["actions"]         # (N,)
    rewards = trajectory_data["rewards"]         # (N,)
    next_states = trajectory_data["next_states"] # (N, 4)
    dones = trajectory_data["dones"]             # (N,)
    init_states = trajectory_data["init_states"] # (N_0, 4)

    N = states.shape[0]

    # Phi(s_t, a_t): (N, d)
    Phi_sa = feature_model.phi(states, actions)
    d = Phi_sa.shape[1]

    # Psi(s_{t+1}): (N, m)  — zero out terminal transitions
    Psi_next = feature_model.psi(next_states)
    m = Psi_next.shape[1]
    mask = (~dones).float().unsqueeze(1)  # (N, 1): 0 for terminal, 1 otherwise

    # ── Psi^T P Phi ≈ (1/N) sum_t psi(s_{t+1}) phi(s_t, a_t)^T  (non-terminal) ──
    # (m, N) @ (N, d) -> (m, d)
    PsiT_P_Phi = (Psi_next * mask).T @ Phi_sa / N

    # ── Psi^T B Phi ≈ (1/N) sum_t psi(s_t) [sum_a phi(s_t, a)]^T ──
    # B operator: (By)(s) = sum_a y(s, a), so Psi^T B Phi = Psi(s)^T [sum_a Phi(s,a)]
    Psi_s = feature_model.psi(states)  # (N, m)
    Phi_sum_a = torch.zeros(N, d, device=states.device, dtype=states.dtype)
    for a in range(n_actions):
        a_vec = torch.full((N,), a, device=states.device, dtype=torch.long)
        Phi_sum_a = Phi_sum_a + feature_model.phi(states, a_vec)
    PsiT_B_Phi = Psi_s.T @ Phi_sum_a / N

    # ── M = Psi^T (B - gamma P) Phi + ridge * I ──
    M = PsiT_B_Phi - gamma * PsiT_P_Phi
    if ridge > 0:
        reg_size = min(M.shape[0], M.shape[1])
        M[:reg_size, :reg_size] += ridge * torch.eye(reg_size, device=M.device, dtype=M.dtype)

    # ── Cost vector: c_j = -(1/N) sum_t r_t phi_j(s_t, a_t) ──
    # Negative because LP is maximize r^T y but reduced form is minimize c^T w
    c = -(Phi_sa * rewards.unsqueeze(1)).sum(dim=0) / N

    # ── RHS: b_i = (1/N_0) sum psi_i(s_0) ──
    b = feature_model.psi(init_states).mean(dim=0)

    return M, c, b


def estimate_lp_matrices_batched(
    feature_model,
    trajectory_data: dict,
    gamma: float = 0.99,
    n_actions: int = 2,
    batch_size: int = 4096,
):
    """Memory-efficient version that processes features in batches.

    Same interface as estimate_lp_matrices but handles large datasets
    without OOM by accumulating in chunks.
    """
    states = trajectory_data["states"]
    actions = trajectory_data["actions"]
    rewards = trajectory_data["rewards"]
    next_states = trajectory_data["next_states"]
    dones = trajectory_data["dones"]
    init_states = trajectory_data["init_states"]

    N = states.shape[0]
    device = states.device

    # Probe feature dimensions
    with torch.no_grad():
        sample_phi = feature_model.phi(states[:1], actions[:1])
        sample_psi = feature_model.psi(states[:1])
    d = sample_phi.shape[1]
    m = sample_psi.shape[1]

    # Accumulators
    PsiT_P_Phi = torch.zeros(m, d, device=device)
    PsiT_B_Phi = torch.zeros(m, d, device=device)
    c_acc = torch.zeros(d, device=device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        s_b = states[start:end]
        a_b = actions[start:end]
        r_b = rewards[start:end]
        ns_b = next_states[start:end]
        d_b = dones[start:end]
        n_b = end - start

        Phi_sa = feature_model.phi(s_b, a_b)
        Psi_next = feature_model.psi(ns_b)
        mask = (~d_b).float().unsqueeze(1)

        PsiT_P_Phi += (Psi_next * mask).T @ Phi_sa

        Psi_s = feature_model.psi(s_b)
        Phi_sum_a = torch.zeros(n_b, d, device=device)
        for act in range(n_actions):
            a_vec = torch.full((n_b,), act, device=device, dtype=torch.long)
            Phi_sum_a += feature_model.phi(s_b, a_vec)
        PsiT_B_Phi += Psi_s.T @ Phi_sum_a

        c_acc += (Phi_sa * r_b.unsqueeze(1)).sum(dim=0)

    M = PsiT_B_Phi / N - gamma * PsiT_P_Phi / N
    c = -c_acc / N
    b = feature_model.psi(init_states).mean(dim=0)

    return M, c, b
