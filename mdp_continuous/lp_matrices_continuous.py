"""
LP matrix estimation for continuous action spaces.

The B operator sum_a phi(s,a) becomes ∫ phi(s,a) da, approximated via
quadrature over the action space.
"""

import torch


def estimate_lp_matrices_continuous(
    feature_model,
    trajectory_data: dict,
    gamma: float = 0.99,
    action_low: float = -3.0,
    action_high: float = 3.0,
    n_quadrature: int = 32,
    ridge: float = 1e-4,
):
    """Estimate M, c, b from trajectory data with continuous actions.

    The key difference from the discrete version is the B operator:
        Psi^T B Phi = Psi(s)^T ∫ Phi(s,a) da
    approximated via uniform quadrature over the action range.

    Args:
        feature_model: object with .phi(states, actions) and .psi(states)
        trajectory_data: dict with states, actions, rewards, next_states, dones, init_states
        gamma: discount factor
        action_low, action_high: action range for quadrature
        n_quadrature: number of quadrature points for action integration
        ridge: diagonal regularization

    Returns:
        M: (m, d) constraint matrix
        c: (d,)   cost vector
        b: (m,)   RHS vector
    """
    states = trajectory_data["states"]           # (N, state_dim)
    actions = trajectory_data["actions"]         # (N,) or (N, action_dim)
    rewards = trajectory_data["rewards"]         # (N,)
    next_states = trajectory_data["next_states"] # (N, state_dim)
    dones = trajectory_data["dones"]             # (N,)
    init_states = trajectory_data["init_states"] # (N_0, state_dim)

    N = states.shape[0]
    device = states.device

    # Phi(s_t, a_t): (N, d)
    Phi_sa = feature_model.phi(states, actions)
    d = Phi_sa.shape[1]

    # Psi(s_{t+1}): (N, m) — zero out terminal transitions
    Psi_next = feature_model.psi(next_states)
    m = Psi_next.shape[1]
    mask = (~dones).float().unsqueeze(1)

    # ── Psi^T P Phi ≈ (1/N) sum_t psi(s_{t+1}) phi(s_t, a_t)^T ──
    PsiT_P_Phi = (Psi_next * mask).T @ Phi_sa / N

    # ── Psi^T B Phi: B operator via quadrature ∫ phi(s,a) da ──
    Psi_s = feature_model.psi(states)  # (N, m)

    # Uniform quadrature points over [action_low, action_high]
    a_quad = torch.linspace(action_low, action_high, n_quadrature, device=device)
    volume = action_high - action_low  # action space volume

    # Accumulate ∫ phi(s, a) da ≈ (volume / K) * sum_k phi(s, a_k)
    Phi_integral = torch.zeros(N, d, device=device, dtype=states.dtype)
    for a_k in a_quad:
        a_vec = torch.full((N,), a_k.item(), device=device, dtype=states.dtype)
        Phi_integral = Phi_integral + feature_model.phi(states, a_vec)
    Phi_integral = Phi_integral * (volume / n_quadrature)

    PsiT_B_Phi = Psi_s.T @ Phi_integral / N

    # ── M = Psi^T (B - gamma P) Phi + ridge * I ──
    M = PsiT_B_Phi - gamma * PsiT_P_Phi
    if ridge > 0:
        m_dim, d_dim = M.shape
        ridge_mat = torch.zeros_like(M)
        for start in range(0, d_dim, m_dim):
            end = min(start + m_dim, d_dim)
            blk = end - start
            ridge_mat[:blk, start:end] = ridge * torch.eye(
                blk, device=M.device, dtype=M.dtype)
        M = M + ridge_mat

    # ── Cost vector: c_j = -(1/N) sum_t r_t phi_j(s_t, a_t) ──
    c = -(Phi_sa * rewards.unsqueeze(1)).sum(dim=0) / N

    # ── RHS: b_i = (1/N_0) sum psi_i(s_0) ──
    b = feature_model.psi(init_states).mean(dim=0)

    return M, c, b
