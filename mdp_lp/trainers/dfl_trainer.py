import torch
import torch.nn as nn
from mdp_lp.mdp import MDP
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver
from mdp_lp.inventory_poisson_grad import dH_dlambda, dr_dlambda


class DFLTrainer:
    def __init__(self, predictor, beta, gamma=None, optimizer=None, use_analytical_grad=False,
                 inventory_params=None):
        self.predictor = predictor
        self.mdp = MDP(beta, gamma)
        self.optimizer = optimizer
        self.use_analytical_grad = use_analytical_grad
        self.inventory_params = inventory_params
        
        if use_analytical_grad and inventory_params is None:
            raise ValueError("inventory_params required when use_analytical_grad=True")

    def forward(self, x):
        P, r = self.predictor(x)
        H, r_flat, gamma = self.mdp.lp_formulation(P, r)
        y_star = DifferentiableLPSolver.apply(H, r_flat, gamma)
        v_star = torch.sum(r_flat * y_star, dim=1)
        return v_star, y_star, P, r, H, r_flat, gamma

    def train_step(self, x, v_true=None):
        if self.optimizer: self.optimizer.zero_grad()

        v_star, y_star, P, r, H, r_flat, gamma = self.forward(x)
        
        if v_true is not None:
            loss = nn.functional.mse_loss(v_star, v_true)
        else:
            loss = -v_star.mean()

        # Original autograd (commented out)
        # loss.backward()

        if self.use_analytical_grad:
            # Predictor outputs lambda when use_analytical_grad=True
            lambda_param_raw = self.predictor(x)
            
            if isinstance(lambda_param_raw, tuple):
                lambda_param = lambda_param_raw[0].squeeze() if len(lambda_param_raw[0].shape) > 1 else lambda_param_raw[0]
            else:
                lambda_param = lambda_param_raw.squeeze() if len(lambda_param_raw.shape) > 1 else lambda_param_raw
            
            if not lambda_param.requires_grad:
                lambda_param = lambda_param.requires_grad_(True)
            
            # Get LP gradients via DifferentiableLPSolver
            H_for_grad = H.clone().detach().requires_grad_(True)
            r_flat_for_grad = r_flat.clone().detach().requires_grad_(True)
            gamma_for_grad = gamma.clone().detach()
            
            y_star_grad = DifferentiableLPSolver.apply(H_for_grad, r_flat_for_grad, gamma_for_grad)
            v_star_grad = torch.sum(r_flat_for_grad * y_star_grad, dim=1)
            
            if v_true is not None:
                loss_for_grad = nn.functional.mse_loss(v_star_grad, v_true)
            else:
                loss_for_grad = -v_star_grad.mean()
            
            loss_for_grad.backward()
            grad_H = H_for_grad.grad  # ∂v/∂H
            grad_r = r_flat_for_grad.grad  # ∂v/∂r
            
            # Analytical gradients (Appendix B)
            b_max = self.inventory_params['b_max']
            s_max = self.inventory_params['s_max']
            a_max = self.inventory_params['a_max']
            p = self.inventory_params.get('p', 1.0)
            h = self.inventory_params.get('h', 1.0)
            
            dH_dlambda_val = dH_dlambda(lambda_param, b_max, s_max, a_max, self.mdp.beta)
            dr_dlambda_val = dr_dlambda(lambda_param, b_max, s_max, a_max, p, h)
            
            # Chain rule: dv/dλ = <∂v/∂H, ∂H/∂λ> + <∂v/∂r, ∂r/∂λ>
            dv_dlambda_H_term = torch.sum(grad_H * dH_dlambda_val, dim=(1, 2))
            dv_dlambda_r_term = torch.sum(grad_r * dr_dlambda_val, dim=1)
            dv_dlambda = dv_dlambda_H_term + dv_dlambda_r_term
            
            if lambda_param.requires_grad:
                lambda_param.backward(gradient=dv_dlambda)
        else:
            loss.backward()

        if self.optimizer: self.optimizer.step()

        return loss.item(), v_star.detach()
