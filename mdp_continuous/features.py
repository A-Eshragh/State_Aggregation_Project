"""
Feature models for continuous-state LP approximation.

State-action features:  phi(s, a) : R^4 x {0,1} -> R^d
State features (Galerkin):  psi(s) = sum_a phi(s, a) : R^4 -> R^d
"""

import torch
import torch.nn as nn
import math


class RandomFourierFeatures(nn.Module):
    """Fixed random features approximating an RBF kernel.

    phi_j(s, a) = sqrt(2/d) * cos(omega_j^T [norm_state; one_hot(a)] + b_j)

    omega ~ N(0, diag(1/bandwidth^2)), b ~ Uniform[0, 2pi]
    """

    def __init__(
        self,
        state_dim: int = 4,
        n_actions: int = 2,
        n_features: int = 256,
        bandwidth: float = 1.0,
        state_scales: torch.Tensor = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_features = n_features
        self.bandwidth = bandwidth

        input_dim = state_dim + n_actions  # [normalized_state; one_hot(action)]

        # Frozen random parameters
        omega = torch.randn(input_dim, n_features) / bandwidth
        bias = torch.rand(n_features) * 2 * math.pi
        self.register_buffer("omega", omega)
        self.register_buffer("bias", bias)

        # State normalization scales (default: CartPole bounds)
        if state_scales is None:
            state_scales = torch.tensor([2.4, 3.0, 0.2095, 3.0])
        self.register_buffer("state_scales", state_scales)

        self.scale = math.sqrt(2.0 / n_features)

    def _encode_input(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Build input vector [normalized_state; one_hot(action)].

        Args:
            states:  (B, 4)
            actions: (B,) int

        Returns:
            x: (B, state_dim + n_actions)
        """
        norm_s = states / self.state_scales  # (B, 4)
        one_hot_a = torch.zeros(states.shape[0], self.n_actions,
                                device=states.device, dtype=states.dtype)
        one_hot_a.scatter_(1, actions.long().unsqueeze(1), 1.0)
        return torch.cat([norm_s, one_hot_a], dim=1)  # (B, 6)

    def phi(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """State-action features phi(s, a).

        Args:
            states:  (B, 4)
            actions: (B,) int

        Returns:
            features: (B, d)
        """
        x = self._encode_input(states, actions)  # (B, input_dim)
        proj = x @ self.omega + self.bias  # (B, d)
        return self.scale * torch.cos(proj)

    def psi(self, states: torch.Tensor) -> torch.Tensor:
        """Galerkin state features: psi(s) = sum_a phi(s, a).

        Args:
            states: (B, 4)

        Returns:
            features: (B, d)
        """
        result = torch.zeros(states.shape[0], self.n_features,
                             device=states.device, dtype=states.dtype)
        for a in range(self.n_actions):
            actions = torch.full((states.shape[0],), a,
                                 device=states.device, dtype=torch.long)
            result = result + self.phi(states, actions)
        return result


class NeuralFeatures(nn.Module):
    """Learnable features parameterized by a neural network.

    phi_xi(s, a) = MLP([normalized_state; one_hot(action)])
    """

    def __init__(
        self,
        state_dim: int = 4,
        n_actions: int = 2,
        n_features: int = 64,
        hidden_dims: list = None,
        state_scales: torch.Tensor = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_features = n_features

        if hidden_dims is None:
            hidden_dims = [128, 128]

        input_dim = state_dim + n_actions

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_features))
        self.net = nn.Sequential(*layers)

        if state_scales is None:
            state_scales = torch.tensor([2.4, 3.0, 0.2095, 3.0])
        self.register_buffer("state_scales", state_scales)

    def _encode_input(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        norm_s = states / self.state_scales
        one_hot_a = torch.zeros(states.shape[0], self.n_actions,
                                device=states.device, dtype=states.dtype)
        one_hot_a.scatter_(1, actions.long().unsqueeze(1), 1.0)
        return torch.cat([norm_s, one_hot_a], dim=1)

    def phi(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """State-action features. (B, 4) x (B,) -> (B, d)"""
        x = self._encode_input(states, actions)
        return self.net(x)

    def psi(self, states: torch.Tensor) -> torch.Tensor:
        """Galerkin state features: psi(s) = sum_a phi(s, a). (B, 4) -> (B, d)"""
        result = torch.zeros(states.shape[0], self.n_features,
                             device=states.device, dtype=states.dtype)
        for a in range(self.n_actions):
            actions = torch.full((states.shape[0],), a,
                                 device=states.device, dtype=torch.long)
            result = result + self.phi(states, actions)
        return result
