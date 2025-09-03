import numpy as np
import warnings
from typing import Dict, Optional, Tuple, Union, Any
from scipy.optimize import minimize
from enum import Enum

from .components import ExogenousComponent, PolynomialComponent, DummySeasonalComponent
from .managers import ParameterManager, MatrixBuilder, StateVectorManager
from .kalman import KalmanFilter, KalmanSmoother


class FitMethod(Enum):
    """Enumeration of available fitting methods."""
    MLE = "mle"
    KALMAN = "kalman"

class ModelState(Enum):
    """Enumeration of model states."""
    UNFIT = "unfit"
    BUILT = "built"
    FITTED = "fitted"

class StateSpaceResults:
    """Container for state space model results."""
    
    def __init__(self, filter_results: Dict, smoothed_results: Dict = None, 
                 fitted_params: np.ndarray = None, method: str = None,
                 optimization_result: Any = None):
        self.filter_results = filter_results
        self.smoothed_results = smoothed_results
        self.fitted_params = fitted_params
        self.method = method
        self.optimization_result = optimization_result
        
    @property
    def loglikelihood(self) -> float:
        """Get the log-likelihood of the fitted model."""
        return self.filter_results.get('loglikelihood', np.nan)
    
    @property
    def aic(self) -> float:
        """Calculate Akaike Information Criterion."""
        if self.fitted_params is None:
            return np.nan
        k = len(self.fitted_params)
        return 2 * k - 2 * self.loglikelihood
    
    @property
    def bic(self) -> float:
        """Calculate Bayesian Information Criterion."""
        if self.fitted_params is None:
            return np.nan
        k = len(self.fitted_params)
        n = len(self.filter_results.get('observations', []))
        return k * np.log(n) - 2 * self.loglikelihood

class StateSpaceModel:
    """
    Main state space model class that orchestrates all components.
    
    This class provides a flexible framework for building and estimating
    state space models with various components like trends, seasonality,
    and exogenous variables.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize StateSpaceModel.
        
        Args:
            verbose: If True, print status messages during fitting
        """
        self.verbose = verbose
        self.state_manager = StateVectorManager()
        self.matrix_builder = None
        self.parameter_manager = None
        self.kalman_filter = None
        self.kalman_smoother = None
        
        # Data
        self.y = None
        self.X = None
        
        # Model state
        self._state = ModelState.UNFIT
        self._results = None
        
        # Model parameters
        self.params = None
        self.transition = None
        
    def add_polynomial(self, order: int = 2, m0: Optional[np.ndarray] = None, 
                      C0: Optional[np.ndarray] = None, discount_factor: float = 0.99,
                      damped: bool = False, name: str = "polynomial") -> 'StateSpaceModel':
        """
        Add polynomial trend component.
        
        Args:
            order: Order of the polynomial (1=random walk, 2=local linear trend, etc.)
            m0: Initial state vector for this component
            C0: Initial state covariance for this component
            damped: Whether to apply damping to higher-order terms
            name: Name identifier for this component
            
        Returns:
            Self for method chaining
        """
        if order < 1:
            raise ValueError("Polynomial order must be >= 1")
            
        component = PolynomialComponent(order=order, m0=m0, C0=C0, damped=damped, discount_factor=discount_factor, name=name)
        self.state_manager.add_component(component)
        self._state = ModelState.UNFIT
        return self
    
    def add_seasonal(self, periods: int, m0: Optional[np.ndarray] = None, 
                    C0: Optional[np.ndarray] = None,  discount_factor: float = 0.99,
                    name: str = "seasonal") -> 'StateSpaceModel':
        """
        Add dummy seasonal component.
        
        Args:
            periods: Number of periods in the seasonal cycle
            m0: Initial state vector for this component
            C0: Initial state covariance for this component
            name: Name identifier for this component
            
        Returns:
            Self for method chaining
        """
        if periods < 2:
            raise ValueError("Seasonal periods must be >= 2")
            
        component = DummySeasonalComponent(periods=periods, m0=m0, C0=C0, name=name)
        self.state_manager.add_component(component)
        self._state = ModelState.UNFIT
        return self
    
    def add_exogenous(self, order: int = 2, m0: Optional[np.ndarray] = None, 
                     C0: Optional[np.ndarray] = None, discount_factor: float = 0.99,
                     damped: bool = False, name: str = "dynamic_regressor") -> 'StateSpaceModel':
        """
        Add exogenous variables component with dynamic coefficients.
        
        Args:
            order: Order of the dynamic coefficient evolution
            m0: Initial state vector for this component
            C0: Initial state covariance for this component
            damped: Whether to apply damping
            name: Name identifier for this component
            
        Returns:
            Self for method chaining
        """
        if order < 1:
            raise ValueError("Exogenous order must be >= 1")
            
        component = ExogenousComponent(
            order=order, 
            m0=m0, 
            C0=C0, 
            damped=damped, 
            discount_factor=discount_factor, 
            name=name
            )
        
        self.state_manager.add_component(component)
        self._state = ModelState.UNFIT
        return self
    
    def build_model(self, params: Optional[np.ndarray] = None) -> 'StateSpaceModel':
        """Build the model matrices and initialize parameters."""
        if len(self.state_manager.components) == 0:
            raise ValueError("No components added to model")
        
        # Initialize managers
        self.matrix_builder = MatrixBuilder(self.state_manager)
        self.parameter_manager = ParameterManager(self.state_manager)
        
        # Get initial parameters from components
        if params is None:
            self.params = self.parameter_manager.get_initial_params(self.y)
        else:
            self.params = params
        
        # Initialize filters
        self.kalman_filter = KalmanFilter(self.matrix_builder, self.parameter_manager)
        self.kalman_smoother = KalmanSmoother(self.kalman_filter)
        
        # Build initial transition matrix
        param_dict = self.parameter_manager.unpack_parameters(self.params)
        self.transition = self.matrix_builder.build_transition_matrix(param_dict)
        
        self._state = ModelState.BUILT
        return self
    
    def set_data(self, y: np.ndarray, X: Optional[np.ndarray] = None) -> 'StateSpaceModel':
        """
        Set the data for the model.
        
        Args:
            y: Observed time series (n_obs,)
            X: Exogenous variables (n_obs, n_vars) or None
            
        Returns:
            Self for method chaining
        """
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be 1-dimensional")
        if len(y) == 0:
            raise ValueError("y cannot be empty")
            
        self.y = y
        
        if X is not None:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.shape[0] != len(y):
                raise ValueError("X and y must have the same number of observations")
        
        self.X = X
        self._state = ModelState.UNFIT if self._state == ModelState.FITTED else self._state
        return self
    
    def _validate_ready_to_fit(self):
        """Validate that the model is ready to be fitted."""
        if self._state == ModelState.UNFIT:
            raise ValueError("Model must be built before fitting. Call build_model() first.")
        if self.y is None:
            raise ValueError("Data must be set before fitting. Call set_data() first.")
    
    def fit_kalman(self, params: Optional[np.ndarray] = None) -> StateSpaceResults:
        """
        Fit the model using Kalman filtering and smoothing with given parameters.
        
        This method performs filtering and smoothing with the provided parameters
        without any parameter optimization.
        
        Args:
            params: Model parameters to use. If None, uses current model parameters.
            
        Returns:
            StateSpaceResults object containing filter and smoother results
        """
        self._validate_ready_to_fit()
        
        if params is not None:
            if len(params) != len(self.params):
                raise ValueError(f"params must have length {len(self.params)}, got {len(params)}")
            self.params = np.array(params)
        
        if self.verbose:
            print("Fitting model using Kalman filter and smoother")
        
        # Run filter
        filter_results = self.kalman_filter.filter(self.y, self.params, self.X)
        
        # Run smoother
        smoothed_results = self.kalman_smoother.smooth(self.y, self.params, self.X)
        
        # Create results object
        self._results = StateSpaceResults(
            filter_results=filter_results,
            smoothed_results=smoothed_results,
            fitted_params=self.params.copy(),
            method=FitMethod.KALMAN.value
        )
        
        self._state = ModelState.FITTED
        
        return self._results
    
    
    def forecast(self, steps: int, X: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecasts for future time periods.
        
        This method extends the state space model beyond the observed data
        to produce multi-step ahead forecasts.
        
        Args:
            steps: Number of steps ahead to forecast
            X: Future values of exogenous variables with shape (steps, n_vars)
               Required if model has exogenous components
            
        Returns:
            Tuple containing:
            - forecast: Array of forecasted values (steps,)
            - forecast_variance: Array of forecast variances (steps,)
            - state_forecast: Array of forecasted state vectors (state_dim, steps)
            - state_covariance: Array of state covariance matrices (state_dim, state_dim, steps)
            
        Raises:
            ValueError: If model not fitted or X dimensions incorrect
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Validate exogenous data if needed
        has_exog = any(isinstance(comp, ExogenousComponent) 
                       for comp in self.state_manager.components)
        
        if has_exog:
            if X is None:
                raise ValueError("X required for model with exogenous components")
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if X.shape[0] != steps:
                raise ValueError(f"X must have {steps} rows")
            if self.X is not None and X.shape[1] != self.X.shape[1]:
                raise ValueError(f"X must have {self.X.shape[1]} columns")
        
        # Get final filtered state and covariance
        final_state = self._results.filter_results['filtered_state'][:, -1]
        final_cov = self._results.filter_results['filtered_covariance'][:, :, -1]
        
        # Get parameters and matrices
        param_dict = self.parameter_manager.unpack_parameters(self.params)
        obs_var = param_dict['observation_variance']
        V = np.array([[obs_var]])
        
        # Get state dimensions
        state_dim = len(final_state)
        
        # Initialize forecast arrays
        forecast = np.zeros(steps)
        forecast_variance = np.zeros(steps)
        state_forecast = np.zeros((state_dim, steps))
        state_covariance = np.zeros((state_dim, state_dim, steps))
        
        # Current state for recursion
        m_t = final_state.copy()
        C_t = final_cov.copy()
        
        P_t = self.transition @ C_t @ self.transition.T
        W_t = self.kalman_filter._compute_evolution_variance(P_t)

        
        for h in range(steps):
            # State prediction (one-step ahead state forecast)
            a_t = self.transition @ m_t
            P_t = self.transition @ C_t @ self.transition.T
            
            # Add evolution variance
            R_t = P_t + W_t
            
            # Store state forecast and covariance for this step
            state_forecast[:, h] = a_t
            state_covariance[:, :, h] = R_t
            
            # Build design matrix for forecast horizon
            t_forecast = len(self.y) + h
            
            # For exogenous components, we need to pass future X values
            if has_exog:
                F_t = self.matrix_builder.build_design_matrix(h, X)
            else:
                F_t = self.matrix_builder.build_design_matrix(t_forecast, None)
            
            F_t = F_t.reshape(1, -1)
            
            # Observation forecast and variance
            f_t = F_t @ a_t
            Q_t = F_t @ R_t @ F_t.T + V
            
            forecast[h] = f_t.item()
            forecast_variance[h] = Q_t.item()
            
            # Update state for next iteration
            m_t = a_t
            C_t = R_t
        
        return forecast, forecast_variance, state_forecast, state_covariance
    
    def fit_mle(self, start_params: Optional[np.ndarray] = None, 
               method: str = 'L-BFGS-B', maxiter: int = 100000, 
               **kwargs) -> StateSpaceResults:
        """
        Fit the model using maximum likelihood estimation.
        
        This method optimizes the model parameters to maximize the likelihood
        of the observed data.
        
        Args:
            start_params: Starting parameters for optimization. If None, uses current params.
            method: Optimization method (default: 'L-BFGS-B')
            maxiter: Maximum number of iterations
            **kwargs: Additional arguments passed to scipy.optimize.minimize
            
        Returns:
            StateSpaceResults object containing optimization and filter results
        """
        self._validate_ready_to_fit()
        
        if start_params is not None:
            if len(start_params) != len(self.params):
                raise ValueError(f"start_params must have length {len(self.params)}, got {len(start_params)}")
            self.params = np.array(start_params)
        
        if self.verbose:
            print("Fitting model using maximum likelihood estimation")
        
        def objective(params):
            """Negative log-likelihood objective function."""
            self.build_model(params)
            result = self.kalman_filter.filter(self.y, params, self.X)
            return -result['loglikelihood']

        
        # Store starting parameters
        start_params_used = self.params.copy()
        
        # Set up optimization options
        opt_options = {'maxiter': maxiter}
        if self.verbose:
            opt_options['disp'] = True

        opt_result = minimize(
            fun=objective,
            x0=self.params,
            method=method,
            options=opt_options,
            **kwargs
        )

        # Handle optimization results
        if opt_result.success:
            self.params = opt_result.x
            if self.verbose:
                print(f"MLE optimization converged in {opt_result.nit} iterations")
        else:
            warnings.warn(f"MLE optimization failed: {opt_result.message}. Using starting parameters.", 
                         UserWarning, stacklevel=2)
            self.params = start_params_used
        
        # Get final results with optimized parameters
        filter_results = self.kalman_filter.filter(self.y, self.params, self.X)
        smoothed_results = self.kalman_smoother.smooth(self.y, self.params, self.X)
        
        # Create results object
        self._results = StateSpaceResults(
            filter_results=filter_results,
            smoothed_results=smoothed_results,
            fitted_params=self.params.copy(),
            method=FitMethod.MLE.value,
            optimization_result=opt_result
        )
        
        self._state = ModelState.FITTED
        
        return self._results
    
    
    @property
    def results(self) -> Optional[StateSpaceResults]:
        """Get the fitting results."""
        return self._results
    
    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._state == ModelState.FITTED
    
    @property
    def is_built(self) -> bool:
        """Check if the model has been built."""
        return self._state in [ModelState.BUILT, ModelState.FITTED]
    
    def get_component_states(self) -> Dict[str, np.ndarray]:
        """
        Extract state vectors for each component after fitting.
        
        Returns:
            Dictionary mapping component names to their state trajectories
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to extract component states")
        
        return self.state_manager.extract_component_states(
            self._results.smoothed_results['states']
        )
    
    def summary(self) -> str:
        """
        Generate a summary of the fitted model.
        
        Returns:
            Formatted string summary of model components and fit statistics
        """
        if not self.is_fitted:
            return "Model not fitted yet"
        
        summary_lines = [
            "State Space Model Summary",
            "=" * 30,
            f"Method: {self._results.method.upper()}",
            f"Observations: {len(self.y)}",
            f"Log-likelihood: {self._results.loglikelihood:.4f}",
            f"AIC: {self._results.aic:.4f}",
            f"BIC: {self._results.bic:.4f}",
            "",
            "Components:",
        ]
        
        for i, component in enumerate(self.state_manager.components):
            summary_lines.append(f"  {i+1}. {component.name} ({type(component).__name__})")
        
        if hasattr(self._results, 'optimization_result') and self._results.optimization_result:
            opt = self._results.optimization_result
            summary_lines.extend([
                "",
                "Optimization:",
                f"  Converged: {opt.success}",
                f"  Iterations: {opt.nit}",
                f"  Function evaluations: {opt.nfev}"
            ])
        
        return "\n".join(summary_lines)
    
    def reset(self) -> 'StateSpaceModel':
        """
        Reset the model to unfit state, keeping components but clearing results.
        
        Returns:
            Self for method chaining
        """
        self._results = None
        self._state = ModelState.BUILT if self.is_built else ModelState.UNFIT
        if self.verbose:
            print("Model reset to unfit state")
        return self