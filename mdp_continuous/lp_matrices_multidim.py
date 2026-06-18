"""
LP matrix estimation for multi-dimensional continuous actions.

B operator: sum_a phi(s,a) → ∫ phi(s,a) da over action space.
Approximated via uniform grid quadrature (tensor product of 1D grids).
"""

import torch


def estimate_lp_matrices_multidim(
    feature_model,
    trajectory_data: dict,
    gamma: float = 0.99,
    action_low=None,       # scalar or (action_dim,) tensor
    action_high=None,
    n_quad_per_dim: int = 8,
    ridge: float = 1e-4,
):
    """Estimate M, c, b for continuous multi-dim actions.

    Quadrature: uniform n_quad_per_dim^action_dim grid over [action_low, action_high].
    For action_dim=2 and n_quad_per_dim=8, this is 64 grid points.

    Returns:
        M: (m, d)
        c: (d,)
        b: (m,)
    """
    states = trajectory_data["states"]
    actions = trajectory_data["actions"]        # (N, action_dim) or (N,)
    rewards = trajectory_data["rewards"]
    next_states = trajectory_data["next_states"]
    dones = trajectory_data["dones"]
    init_states = trajectory_data["init_states"]

    N = states.shape[0]
    device = states.device
    action_dim = feature_model.action_dim

    # Normalize actions shape to (N, action_dim)
    if actions.dim() == 1:
        actions = actions.unsqueeze(1)

    # Normalize action bounds
    if isinstance(action_low, (int, float)):
        action_low = torch.full((action_dim,), float(action_low), device=device)
    if isinstance(action_high, (int, float)):
        action_high = torch.full((action_dim,), float(action_high), device=device)

    # Phi(s_t, a_t)
    Phi_sa = feature_model.phi(states, actions)
    d = Phi_sa.shape[1]

    Psi_next = feature_model.psi(next_states)
    m = Psi_next.shape[1]
    mask = (~dones).float().unsqueeze(1)

    # Psi^T P Phi
    PsiT_P_Phi = (Psi_next * mask).T @ Phi_sa / N

    # Psi^T B Phi via tensor-product quadrature over action space
    Psi_s = feature_model.psi(states)

    # Build grid points
    grid_1d = [torch.linspace(action_low[i].item(), action_high[i].item(),
                              n_quad_per_dim, device=device)
               for i in range(action_dim)]
    meshes = torch.meshgrid(*grid_1d, indexing="ij")
    grid = torch.stack([m_.flatten() for m_ in meshes], dim=1)  # (K, action_dim)
    K = grid.shape[0]

    # Volume per grid cell
    volume = 1.0
    for i in range(action_dim):
        volume *= (action_high[i].item() - action_low[i].item())
    cell_volume = volume / K

    # Accumulate integral
    Phi_integral = torch.zeros(N, d, device=device, dtype=states.dtype)
    for k in range(K):
        a_vec = grid[k].unsqueeze(0).expand(N, -1)  # (N, action_dim)
        Phi_integral = Phi_integral + feature_model.phi(states, a_vec)
    Phi_integral = Phi_integral * cell_volume

    PsiT_B_Phi = Psi_s.T @ Phi_integral / N

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

    c = -(Phi_sa * rewards.unsqueeze(1)).sum(dim=0) / N
    b = feature_model.psi(init_states).mean(dim=0)

    return M, c, b
