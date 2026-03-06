"""
This module contains classes that define the initial conditions for the quantum system.
"""

from typing import Optional

import torch
from torch import nn

from ..math_utils import diff
from .input_normalizer import InputNormalizer


class InitialCondition:
    """
    Class for defining the initial conditions of the quantum system.
    """
    def __init__(self, coordinate: int, coordinate_value, psi_R, psi_I, derivative=False):
        self.coordinate_value = coordinate_value
        self.coordinate = coordinate
        self.psi_R = psi_R
        self.psi_I = psi_I
        self.is_derivative = derivative

    def get_initial_inputs(self, X: torch.Tensor) -> torch.Tensor:
        """
        Get the initial input tensor for the initial condition.
        """
        X_0 = X.clone()
        X_0[:, self.coordinate] = self.coordinate_value
        return X_0
    
    def get_initial_values(self, X_0: torch.Tensor) -> torch.Tensor:
        """
        Get the initial function values for the initial condition.
        """
        raise NotImplementedError("Initial function values must be implemented in subclass.")

    def _derivative(self, model: nn.Module, X_0: torch.Tensor) -> torch.Tensor:
        """
        Calculate the derivative of the function at the initial condition.

        TODO: This method may be incorrect. We may need to calculate the derivative first, and then evaluate it at the initial condition, rather than evaluating the function at the initial condition and then calculating the derivative.
        """
        predicted_values = model(X_0)
        predicted_derivative = diff(predicted_values, X_0, self.coordinate)
        return predicted_derivative

    def loss(self, model: nn.Module, X: torch.Tensor, input_normalizer: Optional[InputNormalizer] = None) -> torch.Tensor:
        """
        Calculate the loss for the initial condition.
        """
        X_0 = self.get_initial_inputs(X)
        initial_values = self.get_initial_values(X_0)
        X_0_model = input_normalizer.normalize(X_0) if input_normalizer is not None else X_0
        if self.is_derivative:
            predicted_derivatives = self._derivative(model, X_0_model)
            return torch.nn.MSELoss()(predicted_derivatives, initial_values)
        predicted_values = model(X_0_model)
        return torch.nn.MSELoss()(predicted_values, initial_values)
