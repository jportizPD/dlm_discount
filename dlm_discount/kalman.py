import numpy as np


import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.linalg import block_diag
import logging
import matplotlib.pyplot as plt

from .components import StateSpaceComponent, ExogenousComponent, PolynomialComponent, DummySeasonalComponent
from .managers import ParameterManager, MatrixBuilder, StateVectorManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# # ============================================================================
# # Kalman Filter
# # ============================================================================

class KalmanFilter:
    """Kalman filter implementation with numerical stability features."""
    
    def __init__(self, matrix_builder: MatrixBuilder, parameter_manager: ParameterManager):
        self.matrix_builder = matrix_builder
        self.parameter_manager = parameter_manager
        
        self.m0, self.C0 = matrix_builder.m0, matrix_builder.C0
    
    def filter(self, y: np.ndarray, params: np.ndarray,
               X: Optional[np.ndarray] = None, 
               store_prior: bool = False) -> Dict[str, Any]:
        """
        Run Kalman filter with discount factors.
        
        Parameters:
        -----------
        y : array-like
            Observations
        params : array-like
            Model parameters
        initial_conditions : InitialConditions
            Initial state and covariance
        exog_data : array-like, optional
            Exogenous variables
        store_prior : bool
            Whether to store prior states for smoothing
            
        Returns:
        --------
        dict : Filter results
        """
        n_obs = len(y)
        if n_obs == 0:
            raise ValueError("No observations provided")
        
        # Unpack parameters
        param_dict = self.parameter_manager.unpack_parameters(params)
        obs_var = param_dict['observation_variance']
        V = np.array([[obs_var]])

        
        # Get system matrices
        T = self.matrix_builder.build_transition_matrix(param_dict)
        dim = T.shape[0]
        
        
        # Initialize storage
        filtered_state = np.zeros((dim, n_obs))
        filtered_cov = np.zeros((dim, dim, n_obs))
        forecasts = np.zeros(n_obs)
        forecast_errors = np.zeros(n_obs)
        forecast_variance = np.zeros(n_obs)
        
        if store_prior:
            prior_state = np.zeros((dim, n_obs))
            prior_cov = np.zeros((dim, dim, n_obs))
        
        # Initialize
        m_t1 = self.m0.copy()
        C_t1 = self.C0.copy()
        loglike = 0.0
        
        for t in range(n_obs):
            # Prior step
            a_t = T @ m_t1
            P_t = T @ C_t1 @ T.T
            
            # Apply discount factors to get evolution variance
            W_t = self._compute_evolution_variance(P_t)
            
            R_t = P_t + W_t
                        
            # Design matrix
            F_t = self.matrix_builder.build_design_matrix(t, X).reshape(1, -1)
            
            # Forecast
            f_t = F_t @ a_t
            Q_t = F_t @ R_t @ F_t.T + V
            Q_t = Q_t.flatten()
            
            # Update
            e_t = y[t] - f_t
            
            A_t = (R_t @ F_t.T) / Q_t
            m_t = a_t + A_t.squeeze() * e_t  # If you want to be explicit about the shape
            I_KH = np.eye(dim) - A_t @ F_t
            C_t = I_KH @ R_t @ I_KH.T + A_t @ V @ A_t.T            

                        
            # Store results
            filtered_state[:, t] = m_t
            filtered_cov[:, :, t] = C_t
            forecasts[t] = f_t
            forecast_errors[t] = e_t
            forecast_variance[t] = Q_t
            
            if store_prior:
                prior_state[:, t] = a_t
                prior_cov[:, :, t] = R_t
            
            # Update log-likelihood
            loglike += -0.5 * (np.log(2 * np.pi) + np.log(Q_t) + e_t**2 / Q_t)
            
            # Update for next iteration
            m_t1 = m_t.copy()
            C_t1 = C_t.copy()
        
        result = {
            'filtered_state': filtered_state,
            'filtered_covariance': filtered_cov,
            'forecasts': forecasts,
            'forecast_errors': forecast_errors,
            'forecast_variance': forecast_variance,
            'loglikelihood': loglike,
            'n_obs': n_obs
        }
        
        if store_prior:
            result['prior_state'] = prior_state
            result['prior_covariance'] = prior_cov
            
        return result
    
    def _compute_evolution_variance(self, P_t: np.ndarray) -> np.ndarray:
        """Compute evolution variance matrix using discount factors."""
        dim = P_t.shape[0]
        W_blocks = []
        
        for component in self.matrix_builder.state_manager.components:
            indices = component.state_indices
            delta = component.discount_factor
            
            # Extract the block for this component
            P_block = P_t[indices, indices]
            
            # Apply discount factor to the entire block
            W_block = ((1 - delta) / delta) * P_block
            W_blocks.append(W_block)
        
        # Combine blocks into full matrix
        W_t = block_diag(*W_blocks) if W_blocks else np.zeros((dim, dim))
        
        return W_t


class KalmanSmoother:
    """Kalman smoother implementation using Rauch-Tung-Striebel recursions."""
    
    def __init__(self, kalman_filter: KalmanFilter):
        self.kalman_filter = kalman_filter
    
    def smooth(self, y: np.ndarray, params: np.ndarray,
               X: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run forward-backward Kalman smoother.
        
        Parameters:
        -----------
        y : array-like
            Observations
        params : array-like
            Model parameters
        initial_conditions : InitialConditions
            Initial state and covariance
        exog_data : array-like, optional
            Exogenous variables
            
        Returns:
        --------
        dict : Smoother results including smoothed states and filter results
        """
        # First run forward filter with prior storage
        filter_results = self.kalman_filter.filter(
            y, params, X, store_prior=True
        )
        param_dict = self.kalman_filter.parameter_manager.unpack_parameters(params)

        
        # Extract needed quantities
        m = filter_results['filtered_state']  # (dim, n_obs)
        C = filter_results['filtered_covariance']  # (dim, dim, n_obs)
        a = filter_results['prior_state']  # (dim, n_obs)
        R = filter_results['prior_covariance']  # (dim, dim, n_obs)
        
        n_obs = filter_results['n_obs']
        dim = m.shape[0]
        
        # Get transition matrix
        T = self.kalman_filter.matrix_builder.build_transition_matrix(param_dict)
        
        # Initialize smoothed estimates
        smoothed_state = np.zeros_like(m)
        smoothed_cov = np.zeros_like(C)
        smoothed_gain = np.zeros((dim, dim, n_obs))
        
        # Initialize backward pass with final filtered estimates
        smoothed_state[:, -1] = m[:, -1]
        smoothed_cov[:, :, -1] = C[:, :, -1]
        
        # Backward pass
        for t in range(n_obs - 2, -1, -1):
            # Compute smoother gain: G_t = C_t * T' * R_{t+1}^{-1}
            try:
                R_inv = np.linalg.inv(R[:, :, t+1])
                G_t = C[:, :, t] @ T.T @ R_inv
            except np.linalg.LinAlgError:
                # Use pseudo-inverse if singular
                R_inv = np.linalg.pinv(R[:, :, t+1])
                G_t = C[:, :, t] @ T.T @ R_inv
            
            # Smoothed state: s_t = m_t + G_t * (s_{t+1} - a_{t+1})
            smoothed_state[:, t] = (
                m[:, t] + G_t @ (smoothed_state[:, t+1] - a[:, t+1])
            )
            
            # Smoothed covariance: S_t = C_t + G_t * (S_{t+1} - R_{t+1}) * G_t'
            smoothed_cov[:, :, t] = (
                C[:, :, t] + 
                G_t @ (smoothed_cov[:, :, t+1] - R[:, :, t+1]) @ G_t.T
            )
            
            smoothed_gain[:, :, t] = G_t
        
        # Compute smoothed forecasts
        smoothed_forecasts = np.zeros(n_obs)
        for t in range(n_obs):
            F_t = self.kalman_filter.matrix_builder.build_design_matrix(t, X)
            smoothed_forecasts[t] = F_t @ smoothed_state[:, t]
        
        # Compute smoothed residuals
        smoothed_residuals = y - smoothed_forecasts
        
        return {
            'smoothed_state': smoothed_state,
            'smoothed_covariance': smoothed_cov,
            'smoothed_gain': smoothed_gain,
            'smoothed_forecasts': smoothed_forecasts,
            'smoothed_residuals': smoothed_residuals,
            'filter_results': filter_results
        }