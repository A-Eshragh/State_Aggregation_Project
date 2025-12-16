import torch
import torch.nn as nn
from mdp_lp.mdp import MDP
from mdp_lp.solvers.differentiable_lp import DifferentiableLPSolver

class DFLTrainer:
    def __init__(self, predictor, beta, gamma=None, optimizer=None):
        self.predictor = predictor
        self.mdp = MDP(beta, gamma)
        self.optimizer = optimizer

    def forward(self, x):
        P, r = self.predictor(x)
        H, r_flat, gamma = self.mdp.lp_formulation(P, r)
        y_star = DifferentiableLPSolver.apply(H, r_flat, gamma)
        v_star = torch.sum(r_flat * y_star, dim=1)
        return v_star, y_star, P, r

    def train_step(self, x, v_true=None):
        if self.optimizer: self.optimizer.zero_grad()

        v_star, _, _, _ = self.forward(x)
        if v_true is not None:
            loss = nn.functional.mse_loss(v_star, v_true)
        else:
            loss = -v_star.mean()

        loss.backward()
        if self.optimizer: self.optimizer.step()

        return loss.item(), v_star.detach()
