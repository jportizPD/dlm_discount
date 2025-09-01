"""
DLM Discount: Dynamic Linear Models with Discount Factors

A flexible framework for building and estimating state space models
with various components like trends, seasonality, and exogenous variables.
"""

__version__ = "0.1.0"
__author__ = "JPOO"
__email__ = "your.email@example.com"

# Import main classes for easy access
from .base import StateSpaceModel, StateSpaceResults, FitMethod, ModelState
from .components import (
    StateSpaceComponent,
    PolynomialComponent, 
    DummySeasonalComponent,
    ExogenousComponent
)
from .kalman import KalmanFilter, KalmanSmoother
from .managers import ParameterManager, MatrixBuilder, StateVectorManager

# Define what gets imported with "from dlm_discount import *"
__all__ = [
    'StateSpaceModel',
    'StateSpaceResults', 
    'FitMethod',
    'ModelState',
    'StateSpaceComponent',
    'PolynomialComponent',
    'DummySeasonalComponent', 
    'ExogenousComponent',
    'KalmanFilter',
    'KalmanSmoother',
    'ParameterManager',
    'MatrixBuilder',
    'StateVectorManager'
]