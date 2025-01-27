import numpy as np
import torch
from nn_mdl import *


class BirthDeath:
    def __init__(self, n, k, feats, pow1, pow2, beta, gt_mdl='exp'):
        self.states = np.arange(1, n + 1)
        self.actions = np.arange(1, k + 1)
        # for faster indexing
        self.state_indices = {state: pos for pos, state in enumerate(self.states)}
        self.action_indices = {action: pos for pos, action in enumerate(self.actions)}
        feats = torch.tensor(feats)
        feats = feats.reshape(1, -1) if feats.ndim == 1 else feats
        self.feats = feats
        self.n_samples, self.n_feats = self.feats.shape
        self.pow1 = pow1
        self.pow2 = pow2
        self.beta = beta
        self.gt_mdl = gt_mdl
        self.trans_probs = self.gen_trans_probs()
        self.obj_fun = self.gen_obj_fun()
        self.gamma = 1 / n * torch.ones(n)

    def gen_trans_probs(self):
        n = len(self.states)
        k = len(self.actions)
        if self.gt_mdl == 'exp':
            assert self.n_feats == 2
            transform_matrix = torch.stack([torch.ones(n - 1), torch.arange(2, n + 1)])
            numerator = torch.repeat_interleave(torch.matmul(self.feats, transform_matrix), repeats=k, dim=1)
            denominator = torch.arange(1, k + 1).repeat(n - 1)
            expon = torch.exp(- numerator / denominator)  # avoid numerical overflow when n is large
            thetas = expon / (expon + 1)
        else:
            mdl = GTNN(self.n_feats, (n - 1) * k)
            thetas = mdl(self.feats)

        theta_array = thetas.reshape(self.n_samples, n - 1, k)
        trans_probs = torch.zeros(self.n_samples, n, n, k)
        trans_probs[:, 0, 1, :] = 1
        trans_probs[:, torch.arange(1, n - 1), torch.arange(2, n), :] = theta_array[:, :-1, :]
        trans_probs[:, -1, -1, :] = theta_array[:, -1, :]
        trans_probs[:, :, 0, :] = 1 - torch.sum(trans_probs[:, :, 1:n, :], dim=2)

        return trans_probs

    def gen_obj_fun(self):
        n = len(self.states)
        k = len(self.actions)
        obj_fun = torch.exp((torch.arange(1, k + 1).reshape(1, -1)) ** self.pow1 /
                            (torch.arange(1, n + 1).reshape(-1, 1)) ** self.pow2)

        return obj_fun
