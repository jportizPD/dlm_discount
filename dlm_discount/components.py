import numpy as np


import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.linalg import block_diag
import logging
import matplotlib.pyplot as plt
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StateSpaceComponent(ABC):
    """Abstract base class for all state space components."""
    
    def __init__(self, name: str):
        self.name = name
        self._dimension = None
        self._state_indices = None
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimension of this component's state vector."""
        pass
    
    @abstractmethod
    def build_transition_block(self) -> np.ndarray:
        """Build the transition matrix block for this component."""
        pass
    
    @abstractmethod
    def build_design_block(self, t: int, exog_data: Optional[np.ndarray] = None) -> np.ndarray:
        """Build the design matrix block for this component at time t."""
        pass
    
    @abstractmethod
    def get_parameter_count(self) -> int:
        """Return number of parameters this component needs."""
        pass
    
    def set_state_indices(self, start_idx: int) -> int:
        """Set the state indices for this component."""
        dim = self.get_dimension()
        self._state_indices = slice(start_idx, start_idx + dim)
        return start_idx + dim
    
    @property
    def state_indices(self) -> slice:
        """Get the state indices for this component."""
        if self._state_indices is None:
            raise ValueError(f"State indices not set for component {self.name}")
        return self._state_indices


# ============================================================================
# Component Implementations
# ============================================================================

class PolynomialComponent(StateSpaceComponent):
    """Polynomial trend component (level, slope, acceleration, etc.) with optional damping."""
    
    def __init__(self, order: int = 2, m0: np.ndarray = None, C0: np.ndarray = None, 
                 damped: bool = False, discount_factor: float = 0.99, 
                 damping_factor: float = 0.95, name: str = "polynomial"):
        super().__init__(name)
        if order < 1:
            raise ValueError("Polynomial order must be at least 1")
        
        if m0 is None:
            self.m0 = np.zeros(order)
        else:
            self.m0 = m0
        
        if C0 is None:
            self.C0 = 1e7 * np.eye(order)
        else:
            self.C0 = C0
            
        self.order = order
        self.damped = damped
        
        # Add discount and damping factor parameters
        self.discount_factor = discount_factor
        self.damping_factor = damping_factor if damped else None
        
        self._dimension = order
        self._transition_matrix = None
        self._current_params = None
    
    def get_dimension(self) -> int:
        return self.order
    
    def build_transition_block(self, damping_factor: Optional[float] = None) -> np.ndarray:
        """
        Build polynomial transition matrix with integration structure and optional damping.
        
        Parameters:
        -----------
        phi : float, optional
            Damping parameter (0 < phi <= 1). Only used if damped=True.
            If None and damped=True, uses the last unpacked parameter value.
        """
        T = np.zeros((self.order, self.order))
        
        if not self.damped:
            # Standard polynomial trend without damping
            np.fill_diagonal(T, 1.0)
            # Upper diagonal for integration
            for i in range(self.order - 1):
                T[i, i + 1] = 1.0
        else:
            # Damped polynomial trend
            if damping_factor is None:
                raise ValueError("Damping parameter mus be provided")
            elif damping_factor < 0 or damping_factor > 1:
                raise ValueError("Damping parameter must be between [0,1]")
           
            if self.order == 1:
                # Simple level with damping
                T[0, 0] = damping_factor
            elif self.order == 2:
                # Level and slope with damping
                T[0, 0] = 1.0
                T[0, 1] = damping_factor
                T[1, 1] = damping_factor
            else:
                # Higher order polynomial with damping
                # Apply damping to the diagonal and integration structure
                T[0, 0] = 1.0  # Level always has unit persistence
                for i in range(1, self.order):
                    T[i, i] = damping_factor  # Damped persistence for higher order terms
                
                # Integration structure with damping
                for i in range(self.order - 1):
                    if i == 0:
                        T[i, i + 1] = damping_factor  # Damped integration from slope to level
                    else:
                        T[i, i + 1] = 1.0  # Standard integration for higher orders
        
        self._transition_matrix = T
        return T
    
    def build_design_block(self, t: int, x: Optional[np.ndarray] = None) -> np.ndarray:
        """Only the level (first state) enters the observation equation."""
        design = np.zeros(self.order)
        design[0] = 1.0
        return design
    
    def get_parameter_count(self) -> int:
        """Return number of parameters this component needs."""
        if self.damped:
            return 1  # One discount factor + one damping factor
        else:
            return 0  # One discount factor only
    
    def update_discount_factor(self, discount_factor: float):
        self.discount_factor = discount_factor
    
    def update_damping_parametert(self, damping_factor: float):
        self.damping_factor = damping_factor
    
    def __repr__(self) -> str:
        damping_str = " (damped)" if self.damped else ""
        return f"PolynomialComponent(order={self.order}, name='{self.name}'{damping_str})"


class DummySeasonalComponent(StateSpaceComponent):
    """Dummy seasonality component where seasonal effects sum to zero.
    
    Note: This component does not support damping as seasonal patterns
    are assumed to be persistent cyclical effects.
    """
    
    def __init__(self, periods: int, m0: np.ndarray = None, C0: np.ndarray = None,
                 discount_factor: float = 0.99, name: str = "seasonal"):
        super().__init__(name)
        if periods < 2:
            raise ValueError("Seasonal periods must be at least 2")
            
        if m0 is None:
            self.m0 = np.zeros(periods - 1)
        else:
            self.m0 = m0
        
        if C0 is None:
            self.C0 = 1e7 * np.eye(periods - 1)
        else:
            self.C0 = C0
            
        self.periods = periods
        self.discount_factor = discount_factor
        self._dimension = periods - 1  # One less due to sum-to-zero constraint
        self._current_params = None
    
    def get_dimension(self) -> int:
        return self.periods - 1
    
    def build_transition_block(self) -> np.ndarray:
        """Build seasonal transition matrix."""
        dim = self.get_dimension()
        T = np.zeros((dim, dim))
        
        # First row: negative sum of all other seasonal states
        T[0, :] = -1.0
        
        # Remaining rows: shift previous seasonal states
        if dim > 1:
            T[1:, :-1] = np.eye(dim - 1)
            
        return T
    
    def build_design_block(self, t: int, x: Optional[np.ndarray] = None) -> np.ndarray:
        """First seasonal state enters the observation equation."""
        design = np.zeros(self.get_dimension())
        if self.get_dimension() > 0:
            design[0] = 1.0
        return design
    
    def get_parameter_count(self) -> int:
        return 0  # One discount factor only (no damping for seasonal)

    def update_discount_factor(self, discount_factor: float):
        self.discount_factor = discount_factor
            
    def __repr__(self) -> str:
        return f"DummySeasonalComponent(periods={self.periods}, name='{self.name}')"


class ExogenousComponent(StateSpaceComponent):
    """Time-varying coefficients for exogenous variables with optional damping."""
    
    def __init__(self, order: int = 2, m0: np.ndarray = None, C0: np.ndarray = None,
                 damped: bool = False, discount_factor: float = 0.99,
                 damping_factor: float = 0.95, name: str = "dynamic_regressor"):
        super().__init__(name)
        if order < 1:
            raise ValueError("Dynamic regressor order must be at least 1")
            
        if m0 is None:
            self.m0 = np.zeros(order)
        else:
            self.m0 = m0
        
        if C0 is None:
            self.C0 = 1e7 * np.eye(order)
        else:
            self.C0 = C0
            
        self.order = order
        self.damped = damped
        
        # Add discount and damping factor parameters
        self.discount_factor = discount_factor
        self.damping_factor = damping_factor if damped else None
        
        self._dimension = order
        self._transition_matrix = None
        self._current_params = None

    def get_dimension(self) -> int:
        return self._dimension
    
    def build_transition_block(self, damping_factor: Optional[float] = None) -> np.ndarray:
        """
        Build polynomial transition matrix with integration structure and optional damping.
        
        Parameters:
        -----------
        phi : float, optional
            Damping parameter (0 < phi <= 1). Only used if damped=True.
            If None and damped=True, uses the last unpacked parameter value.
        """
        T = np.zeros((self.order, self.order))
        
        if not self.damped:
            # Standard polynomial trend without damping
            np.fill_diagonal(T, 1.0)
            # Upper diagonal for integration
            for i in range(self.order - 1):
                T[i, i + 1] = 1.0
        else:
            # Damped polynomial trend
            if damping_factor is None:
                raise ValueError("Damping parameter mus be provided")
            elif damping_factor < 0 or damping_factor > 1:
                raise ValueError("Damping parameter must be between [0,1]")
           
            if self.order == 1:
                # Simple level with damping
                T[0, 0] = damping_factor
            elif self.order == 2:
                # Level and slope with damping
                T[0, 0] = 1.0
                T[0, 1] = damping_factor
                T[1, 1] = damping_factor
            else:
                # Higher order polynomial with damping
                # Apply damping to the diagonal and integration structure
                T[0, 0] = 1.0  # Level always has unit persistence
                for i in range(1, self.order):
                    T[i, i] = damping_factor  # Damped persistence for higher order terms
                
                # Integration structure with damping
                for i in range(self.order - 1):
                    if i == 0:
                        T[i, i + 1] = damping_factor  # Damped integration from slope to level
                    else:
                        T[i, i + 1] = 1.0  # Standard integration for higher orders
        
        self._transition_matrix = T
        return T
    
    def build_design_block(self, t: int, x: Optional[np.ndarray] = None) -> np.ndarray:
        """Build design vector with exogenous values."""
        if x is None:
            raise ValueError("Exogenous data required for ExogenousComponent")
        
        design = np.zeros(self.get_dimension())
        # Only the first state of each variable enters the observation
        design[0] = x[t]
            
        return design
    
    def get_parameter_count(self) -> int:
        """Return number of parameters this component needs."""
        if self.damped:
            return 1  # One discount factor + one damping factor
        else:
            return 0  # One discount factor only
    
    def update_discount_factor(self, discount_factor: float):
        self.discount_factor = discount_factor
    
    def update_damping_parametert(self, damping_factor: float):
        self.damping_factor = damping_factor
    
    def __repr__(self) -> str:
        damping_str = " (damped)" if self.damped else ""
        return f"ExogenousComponent(order={self.order}, name='{self.name}'{damping_str})"