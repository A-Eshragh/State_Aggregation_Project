import random
import numpy as np
import torch
import torch.nn as nn


class GTNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # Ensure outputs are between 0 and 1
        )

    def forward(self, x):
        return self.model(x)


class ThetaPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    # Set Python built-in random seed
    random.seed(seed)

    # Set NumPy seed
    np.random.seed(seed)

    # Set PyTorch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior (PyTorch only)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def expon_mdl(feats, n, k):
    assert feats.ndim == 2 and feats.shape[1] == 2
    transform_matrix = torch.stack([torch.ones(n - 1), torch.arange(2, n + 1)])
    numerator = torch.repeat_interleave(torch.matmul(feats, transform_matrix), repeats=k, dim=1)
    denominator = torch.arange(1, k + 1).repeat(n - 1)
    expon = torch.exp(- numerator / denominator)  # avoid numerical overflow when n is large
    thetas = expon / (expon + 1)

    return thetas
