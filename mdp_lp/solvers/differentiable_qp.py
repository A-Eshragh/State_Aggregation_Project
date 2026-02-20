from re import I
import torch
import torch.autograd as autograd

from .qpth_qp_solver import solve_occupancy_qp_qpth

def extract_basis_from_y(H_b: torch.Tensor, y_b: torch.Tensor, S: int, eps: float = 1e-10) -> torch.Tensor:
    """
    Extract a size-S basis indices B for one sample b, from qpth solution y_b.

    Goal: pick B (len=S) such that:
      - prefer y_b[i] > eps (active)
      - H_b[:, B] is full-rank (rank S), so invertible square matrix

    Args:
        H_b: (S, SA)
        y_b: (SA,)
    Returns:
        B: (S,) long tensor
    """
    SA = y_b.numel()
    device = y_b.device

    # candidate indices sorted by descending y (prefer larger positive mass)
    vals, idx = torch.sort(y_b, descending=True)
    cand = idx[vals > eps]
    if cand.numel() < S:
        # fallback: include zeros too
        cand = idx

    chosen = []
    current = torch.empty((S, 0), device=device, dtype=H_b.dtype)

    def rank_of(mat: torch.Tensor) -> int:
        # mat: (S, k)
        if mat.numel() == 0:
            return 0
        # Use SVD rank with a relative tolerance
        # (torch.linalg.matrix_rank exists, but SVD rank is more controllable)
        u, s, vt = torch.linalg.svd(mat, full_matrices=False)
        tol = s.max() * 1e-10 if s.numel() > 0 else 0.0
        return int((s > tol).sum().item())

    for j in cand.tolist():
        col = H_b[:, j].unsqueeze(1)  # (S,1)
        new = torch.cat([current, col], dim=1)  # (S,k+1)
        if rank_of(new) > rank_of(current):
            chosen.append(j)
            current = new
        if len(chosen) == S:
            break

    if len(chosen) < S:
        # ultimate fallback: just take first S (will likely break if rank-deficient)
        chosen = cand[:S].tolist()

    return torch.tensor(chosen, device=device, dtype=torch.long)


class DifferentiableQPSolver(autograd.Function):
    """
    Differentiable QP layer (basis-restricted) but forward solved by qpth.

    maximize   r^T y - reg * ||y||^2
    s.t.       H y = gamma
               y >= 0

    qpth solves:
      minimize  0.5 y^T Q y + p^T y
      s.t.      G y <= h,  A y = b

    with:
      Q=2*reg*I, p=-r, G=-I, h=0, A=H, b=gamma
    """

    @staticmethod
    def forward(ctx, H, r, gamma, reg: float, qp_solver):
        if reg < 0:
            raise ValueError(f"reg must be >= 0, got {reg}")
        # forward solve with qpth (differentiable, but we still do our own backward)
        with torch.no_grad():
            y_star = solve_occupancy_qp_qpth(H, r, gamma, reg=reg, qp_solver=qp_solver)

            B, S, SA = H.shape
            basis = torch.empty((B, S), device=H.device, dtype=torch.long)
            for b in range(B):
                basis[b] = extract_basis_from_y(H[b], y_star[b], S=S)

        ctx.reg = float(reg)
        # IMPORTANT: qp_solver 不是 tensor，不能 save_for_backward；只存 tensor
        ctx.save_for_backward(H, r, gamma, y_star, basis)
        return y_star

    @staticmethod
    def backward(ctx, grad_y):
        H, r, gamma, y_star, basis = ctx.saved_tensors
        reg = ctx.reg
        batch, S, SA = H.shape

        device = H.device
        dtype = H.dtype

        grad_H = torch.zeros_like(H)
        grad_r = torch.zeros_like(r)
        grad_gamma = torch.zeros_like(gamma)
        grad_reg = None   # 一般不学 reg；要学我可以给你加上
        grad_qp_solver = None

        rho = 2.0 * reg  # 对应 KKT 里的 rho（目标是 min (rho/2)||y||^2 - r^T y）

        if rho <= 0:
            # reg=0 时退化回 LP；固定 basis 的 QP 公式不再适用
            raise RuntimeError("reg=0 makes Q singular; this backward assumes reg>0. "
                               "Use LP layer or add small reg>0.")

        for b in range(batch):
            Bidx = basis[b]            # (S,)
            H_B = H[b, :, Bidx]        # (S,S)
            r_B = r[b, Bidx]           # (S,)
            y_B = y_star[b, Bidx]      # (S,)
            g_B = grad_y[b, Bidx]      # (S,)

            I = torch.eye(S, device=device, dtype=dtype)
            K = torch.zeros((2 * S, 2 * S), device=device, dtype=dtype)
            K[:S, :S] = rho * I
            K[:S, S:] = H_B.T
            K[S:, :S] = H_B

            bvec = torch.cat([r_B, gamma[b]], dim=0)
            z = torch.linalg.solve(K, bvec)  # [y_B; lambda]
            g_z = torch.cat([g_B, torch.zeros(S, device=device, dtype=dtype)], dim=0)
            u = torch.linalg.solve(K.T, g_z)  # K^{-T} g_z

            u_y = u[:S]
            u_lam = u[S:]

            # grads for rhs
            grad_r[b, Bidx] += u_y
            grad_gamma[b] += u_lam

            # grad for K
            grad_K = -torch.outer(u, z)
            grad_H_B = grad_K[S:, :S] + grad_K[:S, S:].T  # (S,S)

            for i, col_idx in enumerate(Bidx):
                grad_H[b, :, col_idx] += grad_H_B[:, i]

        return grad_H, grad_r, grad_gamma, grad_reg, grad_qp_solver


def differentiable_occupancy_qp(H, r, gamma, reg: float, qp_solver):
    return DifferentiableQPSolver.apply(H, r, gamma, reg, qp_solver)