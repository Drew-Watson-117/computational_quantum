"""
This module contains classes that define the initial conditions for the quantum system.
"""

from __future__ import annotations

import torch

from ..math_utils import diff


class InitialCondition:
    """
    Class for defining the initial conditions of the quantum system.
    """

    def __init__(
        self,
        coordinate: int,
        coordinate_value,
        psi_R,
        psi_I,
        derivative: bool = False,
        weight: float = 1.0,
    ):
        self.coordinate_value = coordinate_value
        self.coordinate = coordinate
        self.psi_R = psi_R
        self.psi_I = psi_I
        self.is_derivative = derivative
        self.weight = float(weight)

    def get_initial_inputs(self, X: torch.Tensor) -> torch.Tensor:
        """
        Get the initial input tensor for the initial condition.
        """
        X_0 = X.clone()
        X_0[:, self.coordinate] = self.coordinate_value
        return X_0

    def _coerce_component(self, value, reference: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        if tensor.ndim == 0:
            tensor = tensor.expand(reference.shape[0])
        return tensor

    def get_initial_values(self, X_0: torch.Tensor) -> torch.Tensor:
        """
        Get the initial function values for the initial condition.
        """
        coordinates = [X_0[:, index] for index in range(X_0.shape[1])]
        reference = coordinates[0]
        psi_R = self._coerce_component(self.psi_R(*coordinates), reference)
        psi_I = self._coerce_component(self.psi_I(*coordinates), reference)
        return torch.stack((psi_R, psi_I), dim=1)

    def _derivative(self, model_fn, X_0: torch.Tensor) -> torch.Tensor:
        """
        Calculate component-wise derivatives at the initial condition.
        """
        X_0 = X_0.clone().requires_grad_(True)
        predicted_values = model_fn(X_0)
        predicted_real_derivative = diff(predicted_values[:, 0], X_0, self.coordinate)
        predicted_imag_derivative = diff(predicted_values[:, 1], X_0, self.coordinate)
        return torch.stack((predicted_real_derivative, predicted_imag_derivative), dim=1)

    def loss(self, model_fn, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss for the initial condition.
        """
        X_0 = self.get_initial_inputs(X)
        initial_values = self.get_initial_values(X_0)
        if self.is_derivative:
            predicted_derivatives = self._derivative(model_fn, X_0)
            return self.weight * torch.nn.MSELoss()(predicted_derivatives, initial_values)
        predicted_values = model_fn(X_0)
        return self.weight * torch.nn.MSELoss()(predicted_values, initial_values)
