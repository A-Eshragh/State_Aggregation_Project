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
    """
    Neural network predictor for lambda using price features.
    DGP is linear in log space, so we use price-only inputs (no noise) and a compact MLP.
    """
    def __init__(self, hidden_dim, use_p_related=True, init_mean_lambda=1.0, output_log=False, use_price_only=True):
        """
        Args:
            hidden_dim: Hidden layer dimension (32 recommended for ~100 samples)
            use_p_related: Whether to use p_related as input (exp c)
            init_mean_lambda: Target mean lambda for initialization (default: 1.0)
            output_log: If True, output log(λ); if False, output λ via Softplus (for DFL).
            use_price_only: If True, use only [log(p)] or [log(p), log(p_related)]; drop noise feature.
        """
        super().__init__()
        # Price-only: 1 (exp b) or 2 (exp c) inputs. Matches DGP: ln(λ) = β₀ + β₁·ln(p) + β₂·ln(p_related) + ε
        self.input_dim = (2 if use_p_related else 1) if use_price_only else (3 if use_p_related else 2)
        self.use_p_related = use_p_related
        self.init_mean_lambda = init_mean_lambda
        self.output_log = output_log

        layers = [
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ]
        if not output_log:
            layers.append(nn.Softplus())  # Ensure positive lambda for DFL
        self.net = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        """He init for ReLU; small final-layer scale for stable λ predictions."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        final_linear = self.net[-2] if isinstance(self.net[-1], nn.Softplus) else self.net[-1]
        with torch.no_grad():
            init_mean = max(float(self.init_mean_lambda), 1e-8)
            if isinstance(self.net[-1], nn.Softplus):
                bias_init = torch.log(torch.expm1(torch.tensor(init_mean, dtype=torch.float64))).item()
            else:
                bias_init = torch.log(torch.tensor(init_mean, dtype=torch.float64)).item()
            final_linear.bias.fill_(bias_init)
            final_linear.weight.data *= 0.1

    def forward(self, x):
        """
        Args:
            x: (batch, *) tensor; first columns are [log(p)] or [log(p), log(p_related)].
        Returns:
            If output_log: log(λ) (batch,). If not: λ (batch,).
        """
        x = x[:, :self.input_dim]
        return self.net(x).squeeze(-1)





