import torch
import torch.nn as nn
from mdp_lp.mdp import MDP
from mdp_lp.solvers.dense_lp_solver import solve_lp_gurobi

class TwoStageTrainer:
    def __init__(self, predictor, beta, gamma=None, optimizer=None):
        self.predictor = predictor
        self.mdp = MDP(beta, gamma)
        self.optimizer = optimizer

    def train_step(self, x, P_true, r_true):
        if self.optimizer: self.optimizer.zero_grad()

        P_pred, r_pred = self.predictor(x)
        loss = nn.functional.l1_loss(P_pred, P_true) + \
               nn.functional.l1_loss(r_pred, r_true)

        loss.backward()
        if self.optimizer: self.optimizer.step()

        return loss.item()

    def eval_step(self, x):
        with torch.no_grad():
            P, r = self.predictor(x)
            H, r_flat, gamma = self.mdp.lp_formulation(P, r)
            y_star = solve_lp_gurobi(H, r_flat, gamma)
            v_star = torch.sum(r_flat * y_star, dim=1)
        return v_star
