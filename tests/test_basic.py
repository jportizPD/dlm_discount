import pytest
import numpy as np
from dlm_discount import StateSpaceModel, PolynomialComponent, DummySeasonalComponent

def test_import():
    """Test that main classes can be imported."""
    assert StateSpaceModel is not None
    assert PolynomialComponent is not None
    assert DummySeasonalComponent is not None

def test_basic_model_creation():
    """Test basic model creation and building."""
    model = StateSpaceModel()
    assert model is not None
    assert not model.is_built
    assert not model.is_fitted

def test_polynomial_component():
    """Test polynomial component creation."""
    component = PolynomialComponent(order=2, name="trend")
    assert component.get_dimension() == 2
    assert component.name == "trend"
    assert not component.damped

def test_seasonal_component():
    """Test seasonal component creation."""
    component = DummySeasonalComponent(periods=4, name="quarterly")
    assert component.get_dimension() == 3  # periods - 1
    assert component.name == "quarterly"

def test_model_with_components():
    """Test model with components."""
    np.random.seed(42)
    y = np.random.normal(0, 1, 50)
    
    model = (StateSpaceModel()
             .add_polynomial(order=1, name="level")
             .add_seasonal(periods=4, name="seasonal")
             .set_data(y=y)
             .build_model())
    
    assert model.is_built
# %%
    assert len(model.state_manager.components) == 2


if __name__ == "__main__":
    pytest.main([__file__])
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

    model.summary()
