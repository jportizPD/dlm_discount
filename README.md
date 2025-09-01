# DLM Discount

Dynamic Linear Models with Discount Factors - A flexible framework for building and estimating state space models with various components like trends, seasonality, and exogenous variables.

## Features

- **Modular Components**: Easily combine polynomial trends, seasonal patterns, and dynamic regressors
- **Discount Factors**: Built-in support for discount factor-based evolution variance
- **Damping**: Optional damping for trend and regressor components
- **Kalman Filtering**: Efficient filtering and smoothing implementations
- **MLE Estimation**: Maximum likelihood parameter estimation with optimization
- **Forecasting**: Multi-step ahead forecasting with uncertainty quantification

## Installation

### From Source (Development)

```bash
git clone https://github.com/yourusername/dlm_discount.git
cd dlm_discount
pip install -e .
```

### For Development with Testing Tools

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from dlm_discount import StateSpaceModel

# Generate sample data
np.random.seed(42)
n_obs = 100
t = np.arange(n_obs)
y = 10 + 0.5 * t + 2 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.5, n_obs)

# Create and configure model
model = (StateSpaceModel()
         .add_polynomial(order=2, name="trend")          # Local linear trend
         .add_seasonal(periods=12, name="seasonal")      # Monthly seasonality
         .set_data(y=y)
         .build_model())

# Fit using maximum likelihood estimation
results = model.fit_mle()

# Generate forecasts
forecasts, forecast_vars = model.forecast(steps=24, return_variance=True)

print(model.summary())
```

## Components

### Polynomial Components
- **Order 1**: Random walk (level only)
- **Order 2**: Local linear trend (level + slope)
- **Higher Orders**: Acceleration and beyond
- **Damping**: Optional damping for higher-order terms

### Seasonal Components
- **Dummy Seasonality**: Sum-to-zero seasonal effects
- **Flexible Periods**: Any seasonal cycle length

### Exogenous Components
- **Dynamic Coefficients**: Time-varying regression coefficients
- **Polynomial Evolution**: Coefficients can have trends
- **Multiple Variables**: Support for multiple exogenous series

## Model Building API

The package uses a fluent API for model construction:

```python
model = (StateSpaceModel()
         .add_polynomial(order=2, damped=True)
         .add_seasonal(periods=4)
         .add_exogenous(order=1, name="economic_indicator")
         .set_data(y=observations, X=exogenous_data)
         .build_model())
```

## Fitting Methods

### Kalman Filtering
```python
# Fit with known/fixed parameters
results = model.fit_kalman(params=known_params)
```

### Maximum Likelihood Estimation
```python
# Optimize parameters to maximize likelihood
results = model.fit_mle(method='L-BFGS-B', maxiter=1000)
```

## Results and Diagnostics

Access fitted model results:

```python
# Model fit statistics
print(f"Log-likelihood: {results.loglikelihood}")
print(f"AIC: {results.aic}")
print(f"BIC: {results.bic}")

# Extract component states
component_states = model.get_component_states()
trend_states = component_states['trend']
seasonal_states = component_states['seasonal']

# Model summary
print(model.summary())
```

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0

## License

MIT License - see LICENSE file for details.

## Contributing

This is an internal package. For issues or improvements, please contact the development team.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{dlm_discount,
  title = {DLM Discount: Dynamic Linear Models with Discount Factors},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/dlm_discount}
}
```