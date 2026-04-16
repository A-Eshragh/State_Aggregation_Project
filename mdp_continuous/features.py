"""
Feature models for continuous-state LP approximation.

State-action features:  phi(s, a) : R^4 x {0,1} -> R^d
State features (Galerkin):  psi(s) : R^4 -> R^m   (m < d)

Key design principle: the LP  min c^T w  s.t. Mw = b, w >= 0  must have
d > m so the cost vector c gives optimisation freedom.

For ActionIndexedFeatures (used in DFL), the feature vector is block-diagonal
by action:  phi(s, a=0) = [f(s); 0],  phi(s, a=1) = [0; f(s)].
This gives d = n_actions * k,  m = k,  matching the discrete LP structure
where each (state, action) pair has its own variable.
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
        n_test_features: int = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_features = n_features
        self.bandwidth = bandwidth

        # Test dimension m < d so LP has optimisation freedom
        if n_test_features is None:
            n_test_features = max(1, n_features // 2)
        self.n_test_features = n_test_features

        input_dim = state_dim + n_actions

        # Frozen random parameters
        omega = torch.randn(input_dim, n_features) / bandwidth
        bias = torch.rand(n_features) * 2 * math.pi
        self.register_buffer("omega", omega)
        self.register_buffer("bias", bias)

        # Frozen orthogonal projection: d -> m for Galerkin test functions
        proj = torch.randn(n_features, n_test_features)
        proj, _ = torch.linalg.qr(proj)
        self.register_buffer("test_proj", proj)

        if state_scales is None:
            state_scales = torch.tensor([2.4, 3.0, 0.2095, 3.0])
        self.register_buffer("state_scales", state_scales)

        self.scale = math.sqrt(2.0 / n_features)

    def _encode_input(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        norm_s = states / self.state_scales
        one_hot_a = torch.zeros(states.shape[0], self.n_actions,
                                device=states.device, dtype=states.dtype)
        one_hot_a.scatter_(1, actions.long().unsqueeze(1), 1.0)
        return torch.cat([norm_s, one_hot_a], dim=1)

    def phi(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """State-action features phi(s, a).  (B, 4) x (B,) -> (B, d)"""
        x = self._encode_input(states, actions)
        proj = x @ self.omega + self.bias
        return self.scale * torch.cos(proj)

    def psi(self, states: torch.Tensor) -> torch.Tensor:
        """Galerkin test features: P^T sum_a phi(s, a).  (B, 4) -> (B, m)"""
        result = torch.zeros(states.shape[0], self.n_features,
                             device=states.device, dtype=states.dtype)
        for a in range(self.n_actions):
            actions = torch.full((states.shape[0],), a,
                                 device=states.device, dtype=torch.long)
            result = result + self.phi(states, actions)
        return result @ self.test_proj


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
        n_test_features: int = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_features = n_features

        if n_test_features is None:
            n_test_features = max(1, n_features // 2)
        self.n_test_features = n_test_features

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
        layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

        proj = torch.randn(n_features, n_test_features)
        proj, _ = torch.linalg.qr(proj)
        self.register_buffer("test_proj", proj)

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
        """Galerkin test features: P^T sum_a phi(s, a).  (B, 4) -> (B, m)"""
        result = torch.zeros(states.shape[0], self.n_features,
                             device=states.device, dtype=states.dtype)
        for a in range(self.n_actions):
            actions = torch.full((states.shape[0],), a,
                                 device=states.device, dtype=torch.long)
            result = result + self.phi(states, actions)
        return result @ self.test_proj


class ActionIndexedFeatures(nn.Module):
    """Action-indexed features matching the discrete LP structure.

    phi(s, a=0) = [f(s); 0]      (d = 2k)
    phi(s, a=1) = [0; f(s)]

    where f(s) = softmax(MLP(s) / tau) in R^k (k = d_per_action).

    Using softmax creates localized, soft-partition features — each feature
    activates strongly in a different region of state space. This gives the
    LP meaningful spatial discrimination (similar to soft state aggregation).

    Constraint matrix M has shape (k, 2k):
        M[:, 0:k]   encodes action-0 dynamics
        M[:, k:2k]  encodes action-1 dynamics

    The LP can allocate weight to the action with better dynamics.
    """

    def __init__(
        self,
        state_dim: int = 4,
        n_actions: int = 2,
        n_features: int = 128,
        hidden_dims: list = None,
        state_scales: torch.Tensor = None,
        n_test_features: int = None,  # ignored; always d_per_action
        tau: float = 0.5,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_features = n_features  # total d = n_actions * d_per_action
        self.d_per_action = n_features // n_actions
        self.n_test_features = self.d_per_action  # m = k = d / n_actions
        self.tau = tau

        assert n_features % n_actions == 0, \
            f"n_features ({n_features}) must be divisible by n_actions ({n_actions})"

        if hidden_dims is None:
            hidden_dims = [128, 128]

        # Learnable state network: normalized state -> R^k logits
        # Softmax normalization creates localized features (soft partition of unity).
        input_dim = state_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, self.d_per_action))
        self.state_net = nn.Sequential(*layers)

        if state_scales is None:
            state_scales = torch.tensor([2.4, 3.0, 0.2095, 3.0])
        self.register_buffer("state_scales", state_scales)

    def _state_features(self, states: torch.Tensor) -> torch.Tensor:
        """Base state features f(s).  (B, 4) -> (B, k)

        Softmax normalization creates a soft partition of state space:
        f(s) >= 0 and sum_j f_j(s) = 1. Each feature activates in a
        different state region, giving the LP spatial discrimination.
        """
        norm_s = states / self.state_scales   # (B, 4)
        logits = self.state_net(norm_s)        # (B, k)
        return torch.softmax(logits / self.tau, dim=1)  # (B, k), non-negative, sums to 1

    def phi(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Action-indexed features.  (B, 4) x (B,) -> (B, d)

        phi(s, a) = [0; ...; f(s); ...; 0]  with f(s) in the a-th block.
        """
        B = states.shape[0]
        f_s = self._state_features(states)  # (B, k)

        # one_hot: (B, n_actions), f_s: (B, k)
        # Kronecker: (B, n_actions, k) -> (B, n_actions * k)
        one_hot = torch.zeros(B, self.n_actions,
                              device=states.device, dtype=states.dtype)
        one_hot.scatter_(1, actions.long().unsqueeze(1), 1.0)
        result = (one_hot.unsqueeze(2) * f_s.unsqueeze(1)).reshape(B, -1)
        return result  # (B, d)

    def psi(self, states: torch.Tensor) -> torch.Tensor:
        """Galerkin test features = f(s).  (B, 4) -> (B, k)

        psi(s) = sum_a phi(s,a) summed over action blocks = [f(s); f(s); ...]
        But since all blocks are identical, the test space is just f(s) in R^k.
        No projection needed — m = k = d / n_actions naturally.
        """
        return self._state_features(states)
