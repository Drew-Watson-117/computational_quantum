"""
Subclass of Schrodinger for Cartesian coordinates.
This class implements the necessary methods to compute
the loss for the Schrödinger equation in Cartesian coordinates,
including the initial conditions and the differential equation itself.
The training method is also implemented to optimize the neural network
parameters based on the computed losses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..abstract.schrodinger import Schrodinger
from ..math_utils import diff
from .coordinates import CartesianCoordinates
from .initial_conditions import CartesianInitialCondition


class CartesianSchrodinger(Schrodinger):
    """Class for solving the time-dependent Schrödinger equation in Cartesian coordinates."""

    def __init__(
        self,
        coordinates: list[np.ndarray],
        V,
        initial_conditions: list[CartesianInitialCondition],
        hbar: float = 1.0,
        m: float = 1.0,
        hidden_size: int = 64,
        num_layers: int = 3,
        first_layer_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        """
        Initialize the CartesianSchrodinger class.

        Parameters:
            coordinates: A list of numpy arrays representing the coordinate
                ranges. The first array should be time, followed by spatial
                coordinates (x, y, z).
            V: A function representing the potential energy, which takes the
                appropriate number of spatial coordinates as input.
            initial_conditions: A list of initial-condition instances
                representing the initial conditions for the wavefunction.
            hbar: The reduced Planck constant.
            m: The mass of the particle.
            hidden_size: The number of neurons in each hidden layer of the
                neural network.
            num_layers: The number of hidden layers in the neural network.
            first_layer_omega_0: Frequency scale for the first SIREN layer.
            hidden_omega_0: Frequency scale for the remaining SIREN layers.
        """
        if len(coordinates) < 2:
            raise ValueError("At least two coordinate arrays are required (t and x).")
        if len(coordinates) > 4:
            raise ValueError("No more than four coordinate arrays are allowed (t, x, y, z).")
        self.t = torch.as_tensor(coordinates[CartesianCoordinates.T], dtype=torch.float32)
        self.x = torch.as_tensor(coordinates[CartesianCoordinates.X], dtype=torch.float32)
        self.y = (
            torch.as_tensor(coordinates[CartesianCoordinates.Y], dtype=torch.float32)
            if len(coordinates) > 2
            else None
        )
        self.z = (
            torch.as_tensor(coordinates[CartesianCoordinates.Z], dtype=torch.float32)
            if len(coordinates) > 3
            else None
        )
        super().__init__(
            coordinates,
            V,
            initial_conditions,
            hbar,
            m,
            hidden_size,
            num_layers,
            first_layer_omega_0,
            hidden_omega_0,
        )

    def _create_model_inputs(self) -> torch.Tensor:
        """
        Build the Cartesian product of the coordinate arrays as model inputs.
        """
        coordinates = [self.t, self.x]
        if self.y is not None:
            coordinates.append(self.y)
        if self.z is not None:
            coordinates.append(self.z)
        mesh = torch.meshgrid(*coordinates, indexing="ij")
        return torch.stack([axis.reshape(-1) for axis in mesh], dim=1).to(dtype=torch.float32)

    def get_solution(self) -> pd.DataFrame:
        """
        Get the solution of the Schrödinger equation as a pandas DataFrame.
        """
        if not self.trained:
            raise ValueError("Model must be trained before retrieving solution.")
        psi = self.psi(self.inputs)
        columns = ["t", "x"]
        if self.dimension > 2:
            columns.append("y")
        if self.dimension > 3:
            columns.append("z")
        df = pd.DataFrame(self.inputs.detach().cpu().numpy(), columns=columns)
        df["psi_R"] = psi[:, 0]
        df["psi_I"] = psi[:, 1]
        df["probability_density"] = psi[:, 0] ** 2 + psi[:, 1] ** 2
        return df

    def _dt(self, f, X):
        """
        Calculate the time derivative of the function f with respect to time.
        """
        return diff(f, X, coordinate=CartesianCoordinates.T, order=1)

    def _laplacian(self, psi, X):
        """
        Calculate the spatial Laplacian by summing second derivatives over the
        spatial coordinates.
        """
        laplacian = torch.zeros_like(psi)
        for coordinate in range(1, X.shape[1]):
            laplacian = laplacian + diff(psi, X, coordinate=coordinate, order=2)
        return laplacian

    def _V(self, X):
        """
        Map the components of X to be used in the potential function V.
        """
        if self.dimension == 2:
            value = self.V(X[:, CartesianCoordinates.T], X[:, CartesianCoordinates.X])
        elif self.dimension == 3:
            value = self.V(
                X[:, CartesianCoordinates.T],
                X[:, CartesianCoordinates.X],
                X[:, CartesianCoordinates.Y],
            )
        elif self.dimension == 4:
            value = self.V(
                X[:, CartesianCoordinates.T],
                X[:, CartesianCoordinates.X],
                X[:, CartesianCoordinates.Y],
                X[:, CartesianCoordinates.Z],
            )
        else:
            raise ValueError(f"Invalid number of dimensions. Must be between 2 and 4, but was {self.dimension}")

        value = torch.as_tensor(value, dtype=X.dtype, device=X.device)
        if value.ndim == 0:
            value = value.expand(X.shape[0])
        return value

    def _probability_mass_by_time(self, probability_density: torch.Tensor) -> torch.Tensor:
        """
        Integrate the probability density over the spatial coordinates for each
        time slice.
        """
        grid_shape = [len(self.t), len(self.x)]
        spatial_coordinates = [self.x.to(device=probability_density.device, dtype=probability_density.dtype)]
        if self.y is not None:
            grid_shape.append(len(self.y))
            spatial_coordinates.append(self.y.to(device=probability_density.device, dtype=probability_density.dtype))
        if self.z is not None:
            grid_shape.append(len(self.z))
            spatial_coordinates.append(self.z.to(device=probability_density.device, dtype=probability_density.dtype))

        integrated = probability_density.reshape(*grid_shape)
        for axis, coordinates in reversed(list(enumerate(spatial_coordinates, start=1))):
            integrated = torch.trapz(integrated, x=coordinates, dim=axis)
        return integrated
