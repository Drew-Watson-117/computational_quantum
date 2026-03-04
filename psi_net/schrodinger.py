"""
Base class for solving the time-dependent Schrödinger equation using physics-informed neural networks (PINNs).
"""

from typing import Optional

import pandas as pd
import numpy as np
from torch import nn
import torch
from .initial_conditions import InitialCondition
from .siren import SirenLayer


class Schrodinger:
    """
    Base class defining the interface for solving the time-dependent Schrödinger equation.
    """

    def __init__(
        self,
        V,
        initial_conditions: list[InitialCondition],
        hbar: float = 1.0,
        m: float = 1.0,
        hidden_size: int = 64,
        num_layers: int = 3,
        first_layer_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        """
        Initialize the Schrodinger class.

        Parameters:
            V: A function representing the potential energy as a function of the coordinates.
            initial_conditions: A list of InitialCondition objects defining the initial conditions for the quantum system.
            hbar: The reduced Planck constant (default is 1.0).
            m: The mass of the particle (default is 1.0).
            hidden_size: The number of neurons in each hidden layer of the neural network (default is 64).
            num_layers: The number of hidden layers in the neural network (default is 3).
        """
        self.V = V
        self.initial_conditions = initial_conditions
        self.hbar = hbar
        self.m = m
        self.dimension = None  # To be defined in subclass

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.first_layer_omega_0 = first_layer_omega_0
        self.hidden_omega_0 = hidden_omega_0

        self.inputs = self._create_model_inputs()
        self.model = None  # Defined by _initialize_model in subclass
        self.trained = False

    def _initialize_model(self):
        """
        Initialize the neural network model for approximating the solution to the Schrödinger equation.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)

        # hidden_layers = []
        # for _ in range(self.num_layers):
        #     hidden_layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        #     hidden_layers.append(nn.Tanh())
        # self.model = nn.Sequential(
        #     nn.Linear(self.dimension, self.hidden_size),
        #     *hidden_layers,
        #     nn.Linear(self.hidden_size, 2)
        # ).to(device)

        layers = []

        # First layer
        layers.append(
            SirenLayer(
                self.dimension,
                self.hidden_size,
                is_first=True,
                omega_0=self.first_layer_omega_0,
            )
        )

        # Hidden layers
        for _ in range(self.num_layers):
            layers.append(
                SirenLayer(
                    self.hidden_size,
                    self.hidden_size,
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                )
            )

        # Final linear layer (no sine)
        final_linear = nn.Linear(self.hidden_size, 2)
        with torch.no_grad():
            bound = (6 / self.hidden_size) ** 0.5 / self.hidden_omega_0
            final_linear.weight.uniform_(-bound, bound)

        layers.append(final_linear)

        self.model = nn.Sequential(*layers).to(device)

    def get_coordinate(self, coordinate: int) -> np.array:
        """
        Get the coordinate values for the given coordinate index.

        Parameters:
            coordinate: The index of the coordinate to retrieve.
        Returns:
            A numpy array of shape (num_points,) representing the values of the specified coordinate.
        """
        return self.inputs[:, coordinate].detach().numpy()

    def get_solution(self) -> pd.DataFrame:
        """
        Get the solution of the Schrödinger equation as a pandas DataFrame.

        Returns:
            A DataFrame with columns for each coordinate, the real and imaginary parts of the wavefunction, and the probability density.
        """
        raise NotImplementedError("Solution retrieval must be implemented in subclass.")

    def probability_density(self, X: torch.Tensor) -> np.array:
        """
        Calculate the probability density of the wavefunction at the given input points X.

        Parameters:
            X: A tensor of shape (num_points, dimension) representing the input points.
        Returns:
            A numpy array of shape (num_points,) representing the probability density at the input points.
        """
        if not self.trained:
            raise ValueError(
                "Model must be trained before calculating probability density."
            )
        psi = self.model(X)
        psi_R = psi[:, 0]
        psi_I = psi[:, 1]
        return (psi_R**2 + psi_I**2).detach().numpy()

    def psi(self, X: torch.Tensor) -> np.array:
        """
        Calculate the wavefunction at the given input points X.

        Parameters:
            X: A tensor of shape (num_points, dimension) representing the input points.
        Returns:
            A numpy array of shape (num_points, 2) where the first column is the real part of the wavefunction and the second column is the imaginary part of the wavefunction at the input points.
        """
        if not self.trained:
            raise ValueError("Model must be trained before calculating wavefunction.")
        return self.model(X).detach().numpy()

    def _create_model_inputs(self) -> torch.Tensor:
        """
        Create the input tensor for the neural network based on the coordinate ranges.

        Returns: A tensor of shape (num_points, dimension) where each row is a coordinate point in the input space.
        """
        raise NotImplementedError(
            "Model input creation must be implemented in subclass."
        )

    def _dt(self, f: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """ "
        Calculate the time derivative of the function f.

        Parameters:
            f: A tensor representing the function values at the input points X.
            X: A tensor of shape (num_points, dimension) representing the input points.
        Returns:
            A tensor of shape (num_points,) representing the time derivative of f at the input points.
        """
        raise NotImplementedError("Time derivative must be implemented in subclass.")

    def _laplacian(self, psi: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Laplacian of the wavefunction psi.

        Parameters:
            psi: A tensor of shape (num_points,) representing the wavefunction values at the input points X.
            X: A tensor of shape (num_points, dimension) representing the input points.
        Returns:
            A tensor of shape (num_points,) representing the Laplacian of psi at the input points.
        """
        raise NotImplementedError("Laplacian must be implemented in subclass.")

    def _V(self, X: torch.Tensor) -> torch.Tensor:
        """
        Map the components of X to be used in the potential function V.

        Parameters:
            X: A tensor of shape (num_points, dimension) representing the input points.
        Returns:
            A tensor of shape (num_points,) representing the potential values at the input points.
        """
        raise NotImplementedError("Mapping of V must be implemented in subclass.")

    def ic_loss(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss for the initial conditions.

        Parameters:
            X: A tensor of shape (num_points, dimension) representing the input points.
        Returns:
            A scalar tensor representing the total loss for the initial conditions.
        """
        total_loss = 0.0
        for condition in self.initial_conditions:
            total_loss += condition.loss(self.model, X)
        return total_loss

    def de_loss(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss for the Schrödinger equation.

        Parameters:
            X: A tensor of shape (num_points, dimension) representing the input points.
        Returns:
            A scalar tensor representing the total loss for the Schrödinger equation.
        """
        psi = self.model(X)
        psi_R = psi[:, 0]
        psi_I = psi[:, 1]
        real_part = (
            self.hbar * self._dt(psi_I, X)
            - (self.hbar**2 / (2 * self.m)) * self._laplacian(psi_R, X)
            + self._V(X) * psi_R
        )
        imag_part = (
            -self.hbar * self._dt(psi_R, X)
            - (self.hbar**2 / (2 * self.m)) * self._laplacian(psi_I, X)
            + self._V(X) * psi_I
        )
        return nn.MSELoss()(real_part, torch.zeros_like(real_part)) + nn.MSELoss()(
            imag_part, torch.zeros_like(imag_part)
        )

    def train(
        self,
        num_epochs: int = 1000,
        learning_rate: float = 1e-3,
        de_weight: float = 1.0,
        ic_weight: float = 0.25,
        noise_std: float = 0.01,
        print_every: Optional[int] = 100,
    ):
        """
        Train the neural network to approximate the solution to the Schrödinger equation.

        Parameters:
            num_epochs: The number of training epochs.
            learning_rate: The learning rate for the optimizer.
            de_weight: The weight for the differential equation loss.
            ic_weight: The weight for the initial condition loss.
            noise_std: The standard deviation of the noise added to the inputs during training for better generalization.
            print_every: If provided, the training loss will be printed every 'print_every' epochs.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            X = (
                self.inputs + torch.randn_like(self.inputs) * noise_std
            )  # Add small noise for better training

            loss = ic_weight * self.ic_loss(X) + de_weight * self.de_loss(X)
            loss.backward()
            optimizer.step()

            if print_every and epoch % print_every == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        self.trained = True
