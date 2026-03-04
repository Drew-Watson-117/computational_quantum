"""
Subclass of SchrodingerPINN for Cartesian coordinates.
This class implements the necessary methods to compute
the loss for the Schrödinger equation in Cartesian coordinates,
including the initial conditions and the differential equation itself.
The training method is also implemented to optimize the neural network
parameters based on the computed losses.
"""

from enum import IntEnum
import pandas as pd
import numpy as np
import torch
from math_utils import diff
from initial_conditions import InitialCondition
from schrodinger import Schrodinger

#pylint: disable=invalid-name

class CartesianCoordinates(IntEnum):
    """
    Enumeration for the indices of the coordinates in CartesianSchrodinger.
    """

    T = 0
    X = 1
    Y = 2
    Z = 3


class CartesianInitialCondition(InitialCondition):
    """
    Class for defining the initial conditions of the quantum system in Cartesian coordinates.
    """

    def get_initial_values(self, X_0: torch.Tensor) -> torch.Tensor:
        """
        Get the initial function values for the initial condition in Cartesian coordinates.
        """
        dim = X_0.shape[1]
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
        return torch.stack((psi_R, psi_I), dim=1)

class CartesianSchrodinger(Schrodinger):
    """
    Class for solving the time-dependent Schrödinger equation in Cartesian coordinates.
    """

    def __init__(
        self,
        coordinates: list[np.array],
        V,
        initial_conditions: list[InitialCondition],
        hbar: float = 1.0,
        m: float = 1.0,
        hidden_size: int = 64,
        num_layers: int = 3,
    ):
        """
        Initialize the CartesianSchrodinger class.
        
        Parameters:
            coordinates: A list of numpy arrays representing the coordinate ranges. The first array should be time, followed by spatial coordinates (x, y, z).
            V: A function representing the potential energy, which takes the appropriate number of spatial coordinates as input.
            initial_conditions: A list of InitialCondition or InitialDerivative instances representing the initial conditions for the wavefunction.
            hbar: The reduced Planck constant.
            m: The mass of the particle.
            hidden_size: The number of neurons in each hidden layer of the neural network.
            num_layers: The number of hidden layers in the neural network.
        """
        super().__init__(V, initial_conditions, hbar, m, hidden_size, num_layers)
        if len(coordinates) < 2:
            raise ValueError("At least two coordinate arrays are required.")
        elif len(coordinates) > 4:
            raise ValueError("No more than four coordinate arrays are allowed.")
        self.dimension = len(coordinates)
        self.t = coordinates[CartesianCoordinates.T]
        self.x = coordinates[CartesianCoordinates.X]
        self.y = coordinates[CartesianCoordinates.Y] if len(coordinates) > 2 else None
        self.z = coordinates[CartesianCoordinates.Z] if len(coordinates) > 3 else None
        self.inputs = self._create_model_inputs()
        self._initialize_model()

    def _create_model_inputs(self) -> torch.Tensor:
        all_sets = []
        if self.dimension == 2:
            for _t in self.t:
                for _x in self.x:
                    all_sets.append([_t, _x])
        elif self.dimension == 3:
            for _t in self.t:
                for _x in self.x:
                    for _y in self.y:
                        all_sets.append([_t, _x, _y])
        elif self.dimension == 4:
            for _t in self.t:
                for _x in self.x:
                    for _y in self.y:
                        for _z in self.z:
                            all_sets.append([_t, _x, _y, _z])
        return torch.tensor(all_sets, dtype=torch.float32, requires_grad=True)

    def get_solution(self) -> pd.DataFrame:
        """
        Get the solution of the Schrödinger equation as a pandas DataFrame.
        """
        if not self.trained:
            raise ValueError("Model must be trained before retrieving solution.")
        psi = self.model(self.inputs)
        psi_R = psi[:, 0].detach().numpy()
        psi_I = psi[:, 1].detach().numpy()
        columns = ['t', 'x']
        if self.dimension > 2:
            columns.append('y')
        if self.dimension > 3:
            columns.append('z')
        df = pd.DataFrame(self.inputs.detach().numpy(), columns=columns)
        df['psi_R'] = psi_R
        df['psi_I'] = psi_I
        df['probability_density'] = psi_R**2 + psi_I**2
        return df
        

    def _dt(self, f, X):
        """
        Calculate the time derivative of the function f with respect to time.
        """
        return diff(f, X, coordinate=CartesianCoordinates.T, order=1)

    def _laplacian(self, psi, X):
        laplacian = torch.zeros_like(psi)
        for i in range(1, X.shape[1]):
            laplacian += diff(psi, X, coordinate=i, order=2)
        return laplacian

    def _V(self, X):
        """
        Map the components of X to be used in the potential function V.
        """
        if self.dimension == 2:
            return self.V(X[:, CartesianCoordinates.T], X[:, CartesianCoordinates.X])
        elif self.dimension == 3:
            return self.V(
                X[:, CartesianCoordinates.T],
                X[:, CartesianCoordinates.X],
                X[:, CartesianCoordinates.Y],
            )
        elif self.dimension == 4:
            return self.V(
                X[:, CartesianCoordinates.T],
                X[:, CartesianCoordinates.X],
                X[:, CartesianCoordinates.Y],
                X[:, CartesianCoordinates.Z],
            )
