"""Optimization layer — HPO and NAS for the world model.

Two regimes, chosen by parameter type:

- **Continuous** (crop traits, soil params, residual weights): first-order
  optimization *through the differentiable rollout* — variational
  assimilation. No black-box sampler can match gradient information here.
- **Discrete / structural** (architecture, hidden sizes, feature choices):
  Sobol coverage + successive halving, or Optuna TPE/NSGA-II when installed.
"""
