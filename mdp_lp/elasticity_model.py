"""
Elasticity model for demand parameter lambda in inventory optimization.

ln(λ) = β₀ + β₁ ln(p) + β₂ ln(p_related) + ε
where ε ~ N(0, σ²)
"""

import torch
import torch.nn as nn


class ElasticityModel:
    """Ground truth elasticity model for generating true lambda values."""
    def __init__(self, beta_0, beta_1, beta_2, sigma_epsilon=0.1):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.sigma_epsilon = sigma_epsilon
    
    def __call__(self, p, p_related=None):
        """
        Generate lambda from elasticity model.
        
        Args:
            p: (batch,) tensor of selling prices
            p_related: (batch,) tensor of related item prices (optional)
            
        Returns:
            lambda: (batch,) tensor of Poisson rate parameters
        """
        log_p = torch.log(p + 1e-10)
        log_lambda = self.beta_0 + self.beta_1 * log_p
        
        if p_related is not None and self.beta_2 != 0:
            log_p_related = torch.log(p_related + 1e-10)
            log_lambda = log_lambda + self.beta_2 * log_p_related
        
        epsilon = torch.randn_like(log_p) * self.sigma_epsilon
        log_lambda = log_lambda + epsilon
        
        lambda_val = torch.exp(log_lambda)
        return lambda_val


class LambdaPredictor(nn.Module):
    """Neural network predictor for lambda using price features."""
    def __init__(self, hidden_dim, use_p_related=True, init_mean_lambda=1.0):
        """
        Args:
            hidden_dim: Hidden layer dimension
            use_p_related: Whether to use p_related as input
            init_mean_lambda: Target mean lambda for initialization (default: 1.0)
        """
        super().__init__()
        input_dim = 3 if use_p_related else 2  # [p, p_related, noise] or [p, noise]
        self.use_p_related = use_p_related
        self.init_mean_lambda = init_mean_lambda
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive lambda
        )
        
        # Initialize weights to produce reasonable lambda values
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to produce reasonable lambda predictions."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        final_layer = self.net[-2]
        with torch.no_grad():
            final_layer.bias.fill_(torch.log(torch.tensor(self.init_mean_lambda)).item())
            final_layer.weight.data *= 0.1
    
    def forward(self, x):
        """
        Args:
            x: (batch, input_dim) tensor where:
               - x[:, 0] = p (selling price)
               - x[:, 1] = p_related (if use_p_related) or noise
               - x[:, 2] = noise (if use_p_related)
        Returns:
            lambda: (batch,) tensor of predicted Poisson rates
        """
        lambda_val = self.net(x).squeeze(-1)
        return lambda_val





