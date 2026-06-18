"""
mdp_lp package.

Keep imports lightweight so utility/debug modules can run without optional deps
(e.g. qpth used by some training variants).
"""

__all__ = []

# Intentionally avoid importing heavy modules (e.g. train_elasticity) at import-time.
# Import those explicitly from their modules when needed:
#   from mdp_lp.train_elasticity import train_eval_elasticity