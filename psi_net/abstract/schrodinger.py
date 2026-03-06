"""
Base class for solving the time-dependent Schrödinger equation using physics-informed neural networks (PINNs).
"""

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
import numpy as np
from torch import nn
import torch
from .initial_conditions import InitialCondition
from .siren import SirenLayer
from .input_normalizer import InputNormalizer


class Schrodinger(ABC):
    """
    Base class defining the interface for solving the time-dependent Schrödinger equation.
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
        first_layer_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
    ):
        """
        Initialize the Schrodinger class.

        Parameters:
            coordinates: A list of numpy arrays representing the coordinate ranges.
            V: A function representing the potential energy as a function of the coordinates.
            initial_conditions: A list of InitialCondition objects defining the initial conditions for the quantum system.
            hbar: The reduced Planck constant (default is 1.0).
            m: The mass of the particle (default is 1.0).
            hidden_size: The number of neurons in each hidden layer of the neural network (default is 64).
            num_layers: The number of hidden layers in the neural network (default is 3).
        """
        self.V = V
        self.hbar = hbar
        self.m = m
        self.dimension = len(coordinates)
        self.initial_conditions = initial_conditions

        self.inputs = self._create_model_inputs()
        self.input_normalizer = InputNormalizer(self.inputs)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.first_layer_omega_0 = first_layer_omega_0
        self.hidden_omega_0 = hidden_omega_0

        self.model = self._initialize_model()
        self.trained = False



    def _initialize_model(self) -> nn.Sequential:
        """
        Initialize the neural network model for approximating the solution to the Schrödinger equation.

        Returns:
            An instance of nn.Sequential representing the neural network model.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", device)

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

        return nn.Sequential(*layers).to(device)

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
        psi = self.psi(X)
        psi_R = psi[:, 0]
        psi_I = psi[:, 1]
        return psi_R**2 + psi_I**2

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
        X_normalized = self.input_normalizer.normalize(X)
        return self.model(X_normalized).detach().numpy()

    @abstractmethod
    def _create_model_inputs(self) -> torch.Tensor:
        """
        Create the input tensor for the neural network based on the coordinate ranges.

        Returns: A tensor of shape (num_points, dimension) where each row is a coordinate point in the input space.
        """
        raise NotImplementedError(
            "Model input creation must be implemented in subclass."
        )

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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
            total_loss += condition.loss(self.model, X, self.input_normalizer)
        return total_loss / (total_loss.detach() + 1e-8)  # Normalize to prevent scale issues
    
    def normalization_loss(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate a loss term to encourage the model to produce normalized wavefunctions.

        Parameters:
            X: A tensor of shape (num_points, dimension) representing the input points.
        Returns:
            A scalar tensor representing the normalization loss.
        """
        psi = self.model(X)
        psi_R = psi[:, 0]
        psi_I = psi[:, 1]
        probability_density = psi_R**2 + psi_I**2
        normalization_loss = torch.mean((probability_density - 1.0) ** 2)
        return normalization_loss / (normalization_loss.detach() + 1e-8)  # Normalize to prevent scale issues

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
        total_loss =  nn.MSELoss()(real_part, torch.zeros_like(real_part)) + nn.MSELoss()(
            imag_part, torch.zeros_like(imag_part)
        )
        return total_loss / (total_loss.detach() + 1e-8)  # Normalize to prevent scale issues

    def train(
        self,
        num_epochs: int = 1000,
        learning_rate: float = 1e-3,
        frac_epochs_ic_only: float = 0.4,
        frac_epochs_to_fully_weight_de: float = 0.65,
        de_weight: float = 0.7,
        ic_weight: float = 1.0,
        norm_weight: float = 0.1,
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
        self.model.train()
        remaining_epochs = self.train_phase_1(
            total_epochs=num_epochs, 
            ic_weight=ic_weight,
            norm_weight=norm_weight,
            frac_epochs=frac_epochs_ic_only, 
            learning_rate=learning_rate, 
            noise_std=noise_std, 
            print_every=print_every
        )
        num_epochs_to_fully_weight_de = max(num_epochs * frac_epochs_to_fully_weight_de - (num_epochs - remaining_epochs), 0)
        self.train_phase_2(
            num_epochs=remaining_epochs,
            learning_rate=learning_rate,
            de_weight=de_weight,
            ic_weight=ic_weight,
            norm_weight=norm_weight,
            num_epochs_to_fully_weight_de=num_epochs_to_fully_weight_de,
            noise_std=noise_std,
            print_every=print_every
        )

        self.trained = True
        self.model.eval()

    def train_phase_1(
        self,
        total_epochs: int,
        ic_weight: float = 1.0,
        norm_weight: float = 1.0,
        frac_epochs: float = 0.4,
        learning_rate: float = 1e-3,
        noise_std: float = 0.01,
        print_every: Optional[int] = 100,
    ) -> int:
        """
        Train the model for the initial conditions and normalization only.

        Parameters:
            total_epochs: The total number of epochs to train.
            frac_epochs: The fraction of epochs to use for initial conditions training.
            learning_rate: The learning rate for the optimizer.
            noise_std: The standard deviation of the noise added to the inputs during training.
            print_every: If provided, the training loss will be printed every 'print_every' epochs.
        Returns:
            Number of remaining epochs
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        num_epochs = int(total_epochs * frac_epochs)
        print(f"Training for initial conditions and normalization only for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            X = (
                self.inputs + torch.randn_like(self.inputs) * noise_std
            )  # Add small noise for better training
            ic_loss = ic_weight * self.ic_loss(X)
            X_normalized = self.input_normalizer.normalize(X)
            norm_loss = norm_weight * self.normalization_loss(X_normalized)
            loss = ic_loss + norm_loss
            loss.backward(retain_graph=True)
            optimizer.step()
            if print_every and epoch % print_every == 0:
                print(f"Epoch {epoch}, IC Loss: {ic_loss.item()}, Norm Loss: {norm_loss.item()}, Total Loss: {loss.item()}")
        return total_epochs - num_epochs
    
    def train_phase_2(
        self,
        num_epochs: int,
        de_weight: float,
        ic_weight: float,
        norm_weight: float,
        num_epochs_to_fully_weight_de: int = 500,
        noise_std: float = 0.01,
        learning_rate: float = 1e-3,
        print_every: Optional[int] = 100
    ):
        """
        Train the model for both the initial conditions and the differential equation.

        Parameters:
            num_epochs: The number of epochs to train.
            learning_rate: The learning rate for the optimizer.
            de_weight: The weight for the differential equation loss.
            ic_weight: The weight for the initial condition loss.
            noise_std: The standard deviation of the noise added to the inputs during training.
            print_every: If provided, the training loss will be printed every 'print_every' epochs.
        """
        print(f"Training for both initial conditions and differential equation for remaining {num_epochs} epochs...")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            X = (
                self.inputs + torch.randn_like(self.inputs) * noise_std
            )  # Add small noise for better training
            X_normalized = self.input_normalizer.normalize(X)
            de_weight_adjusted = min(
                de_weight, 
                de_weight * epoch / (num_epochs_to_fully_weight_de + 1e-8)
            )  # Increase DE weight over time
            ic_loss = ic_weight * self.ic_loss(X)
            de_loss = de_weight_adjusted * self.de_loss(X_normalized)
            norm_loss = norm_weight * self.normalization_loss(X_normalized)
            loss = ic_loss + de_loss + norm_loss
            loss.backward(retain_graph=True)
            optimizer.step()

            if print_every and epoch % print_every == 0:
                print(f"Epoch {epoch}, IC Loss: {ic_loss.item()}, DE Loss: {de_loss.item()}, Norm Loss: {norm_loss.item()}, Total Loss: {loss.item()}")
