import numpy as np

from typing import Dict, List, Optional, Any
from scipy.linalg import block_diag
import logging

logger = logging.getLogger(__name__)

class ParameterManager:
    """Manages parameter packing/unpacking across components."""
    
    def __init__(self, state_manager):
        self.state_manager = state_manager
        self._parameter_count = None
        self._parameter_map = None
        self._build_parameter_map()
    
    def _build_parameter_map(self):
        """Build a map of parameter indices for each component."""
        self._parameter_map = {}
        idx = 0
        
        # Observation variance comes first
        self._parameter_map['observation_variance'] = idx
        idx += 1
        
        # Map parameters for each component
        for component in self.state_manager.components:
            param_count = component.get_parameter_count()
            self._parameter_map[component.name] = {
                'start': idx,
                'end': idx + param_count,
                'count': param_count
            }
            idx += param_count
        
        self._parameter_count = idx
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        if self._parameter_count is None:
            self._build_parameter_map()
        return self._parameter_count
    
    def get_initial_params(self, y) -> np.ndarray:
        """Get initial parameter values from components."""
        params = np.zeros(self.get_parameter_count())
        
        # Default observation variance (log scale)
        params[0] = np.log(np.var(y))
        
        # Get initial values from each component
        for component in self.state_manager.components:
            comp_info = self._parameter_map[component.name]
            start_idx = comp_info['start']
            
            # Pack damping factor if present
            if hasattr(component, 'damped') and component.damped:
                if hasattr(component, 'damping_factor'):
                    phi = component.damping_factor
                    phi_scaled = (phi - 0.9) / 0.1

                    params[start_idx] = np.arctanh(phi_scaled)
        
        return params
    
    def unpack_parameters(self, params: np.ndarray) -> Dict[str, Any]:
        """Unpack parameter vector into component parameters."""
        if len(params) != self.get_parameter_count():
            raise ValueError(f"Expected {self.get_parameter_count()} parameters, got {len(params)}")
        
        result = {}
        
        # Observation variance (exp transform for positivity)
        result['observation_variance'] = np.exp(params[0])
        
        # Component parameters
        for component in self.state_manager.components:
            comp_info = self._parameter_map[component.name]
            start_idx = comp_info['start']
            end_idx = comp_info['end']
            comp_params_raw = params[start_idx:end_idx]
            
            # Unpack based on component structure
            comp_params = {}

            # Check if component has damping
            if hasattr(component, 'damped') and component.damped:
                raw_phi = comp_params_raw[0]
                comp_params['damping_factor'] = 0.9 + 0.1 * np.tanh(raw_phi) 
            
            result[component.name] = comp_params
        
        return result
    
    def pack_parameters(self, param_dict: Dict[str, Any]) -> np.ndarray:
        """Pack parameter dictionary into parameter vector."""
        params = np.zeros(self.get_parameter_count())
        
        # Observation variance
        params[0] = np.log(param_dict['observation_variance'])
        
        # Component parameters
        for component in self.state_manager.components:
            comp_info = self._parameter_map[component.name]
            start_idx = comp_info['start']
            comp_params = param_dict[component.name]
            
            # Pack discount factor
            delta = component.discount_factor
            delta_scaled = np.clip((delta - 0.95) / 0.05, -0.999, 0.999)
            params[start_idx] = np.arctanh(delta_scaled)
            
            # Pack damping factor if present
            if hasattr(component, 'damped') and component.damped:
                if hasattr(component, 'damping_factor'):
                    phi = component.damping_factor
                    phi_scaled = np.clip((phi - 0.9) / 0.1, -0.999, 0.999)
                    params[start_idx + 1] = np.arctanh(phi_scaled)
        return params
    
    def get_parameter_bounds(self) -> List[tuple]:
        """Get bounds for optimization (in transformed space)."""
        bounds = []
        
        # Observation variance: positive, so no bounds in log space
        bounds.append((None, None))
        
        # Component parameters
        for component in self.state_manager.components:
            # Discount factor: (0, 1) -> no bounds in logit space
            bounds.append((None, None))
            
            # Damping factor if present
            if hasattr(component, 'damped') and component.damped:
                bounds.append((None, None))
        
        return bounds
    
    def get_parameter_names(self) -> List[str]:
        """Get descriptive names for all parameters."""
        names = ['log_obs_variance']
        
        for component in self.state_manager.components:
            names.append(f'{component.name}_discount')
            if hasattr(component, 'damped') and component.damped:
                names.append(f'{component.name}_damping')
        
        return names


class MatrixBuilder:
    """Constructs system matrices from components."""
    
    def __init__(self, state_manager):
        self.state_manager = state_manager
        self.m0, self.C0 = self.build_prior_states()
    
    def build_prior_states(self) -> tuple:
        """Build the prior mean state m0 and covariance matrix C0."""
        dim = self.state_manager.total_dimension
        
        m0 = np.zeros(dim)
        C0_list = []
        
        for component in self.state_manager.components:
            indices = component.state_indices
            m0[indices] = component.m0
            C0_list.append(component.C0)

        C0 = block_diag(*C0_list) if C0_list else np.zeros((dim, dim))
        
        return m0, C0
    
    def build_transition_matrix(self, param_dict: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Build the complete transition matrix."""
        dim = self.state_manager.total_dimension
        if dim == 0:
            return np.zeros((0, 0))
        
        T = np.zeros((dim, dim))
        
        for component in self.state_manager.components:
            indices = component.state_indices
            
            # Get transition block with optional damping
            if param_dict and component.name in param_dict:
                component_params = param_dict[component.name]
                if 'damping_factor' in component_params:
                    T_block = component.build_transition_block(
                        damping_factor=component_params['damping_factor']
                    )
                else:
                    T_block = component.build_transition_block()
            else:
                T_block = component.build_transition_block()
            
            T[indices, indices] = T_block
            
        return T
    
    def build_design_matrix(self, t: int, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Build the design matrix for time t."""
        dim = self.state_manager.total_dimension
        if dim == 0:
            return np.zeros(0)
        
        F = np.zeros((1, dim))
        x_index = 0
        for component in self.state_manager.components:
            indices = component.state_indices
            
            # Check if this is an exogenous component
            if component.__class__.__name__ == 'ExogenousComponent':
                if X is None:
                    raise ValueError(f"Exogenous data required for component '{component.name}'")
                
                x = X[:, x_index]
                F_block = component.build_design_block(t, x)
                x_index += 1

            else:
                # Non-exogenous component
                F_block = component.build_design_block(t)
            
            F[0, indices] = F_block
            
        return F


class StateVectorManager:
    """Manages state vector organization across components."""
    
    def __init__(self):
        self.components: List = []
        self._total_dimension = 0
        self._component_map = {}
    
    def add_component(self, component) -> None:
        """Add a component and assign its state indices."""
        if component.name in self._component_map:
            raise ValueError(f"Component '{component.name}' already exists")
        
        next_idx = component.set_state_indices(self._total_dimension)
        self.components.append(component)
        self._component_map[component.name] = component
        self._total_dimension = next_idx
    
    def get_component(self, name: str):
        """Get component by name."""
        if name not in self._component_map:
            raise ValueError(f"Component '{name}' not found")
        return self._component_map[name]
    
    @property
    def total_dimension(self) -> int:
        return self._total_dimension
    
    def get_state_slice(self, component_name: str) -> slice:
        """Get state indices for a specific component."""
        return self.get_component(component_name).state_indices
    
    def extract_component_states(self, states: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract state vectors for each component from full state matrix."""
        result = {}
        for component in self.components:
            indices = component.state_indices
            result[component.name] = states[indices, :]
        return result