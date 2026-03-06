"""
Module containing the CartesianInitialCondition class for defining the initial conditions of the quantum system in Cartesian coordinates.
"""

import torch
from ..abstract.initial_conditions import InitialCondition
from .coordinates import CartesianCoordinates

class CartesianInitialCondition(InitialCondition):
    """
    Class for defining the initial conditions of the quantum system in Cartesian coordinates.
    """

    def get_initial_values(self, X_0: torch.Tensor) -> torch.Tensor:
        """
        Get the initial function values for the initial condition in Cartesian coordinates.
        """
        dim = X_0.shape[1]
        psi_R = None
        psi_I = None
        if dim == 2:
            psi_R = self.psi_R(X_0[:, CartesianCoordinates.T], X_0[:, CartesianCoordinates.X])
            psi_I = self.psi_I(X_0[:, CartesianCoordinates.T], X_0[:, CartesianCoordinates.X])
        elif dim == 3:
            psi_R = self.psi_R(
                X_0[:, CartesianCoordinates.T], X_0[:, CartesianCoordinates.X], X_0[:, CartesianCoordinates.Y]
            )
            psi_I = self.psi_I(
                X_0[:, CartesianCoordinates.T], X_0[:, CartesianCoordinates.X], X_0[:, CartesianCoordinates.Y]
            )
        elif dim == 4:
            psi_R = self.psi_R(
                X_0[:, CartesianCoordinates.T],
                X_0[:, CartesianCoordinates.X],
                X_0[:, CartesianCoordinates.Y],
                X_0[:, CartesianCoordinates.Z],
            )
            psi_I = self.psi_I(
                X_0[:, CartesianCoordinates.T],
                X_0[:, CartesianCoordinates.X],
                X_0[:, CartesianCoordinates.Y],
                X_0[:, CartesianCoordinates.Z],
            )
        else:
            raise ValueError(f"Invalid number of dimensions. Must be between 2 and 4, but was {dim}")
        return torch.stack((psi_R, psi_I), dim=1)
