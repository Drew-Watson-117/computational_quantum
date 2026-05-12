"""
Base class for solving the time-dependent Schrödinger equation using
physics-informed neural networks (PINNs).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from typing import Optional
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn

from .initial_conditions import InitialCondition
from .input_normalizer import InputNormalizer
from .siren import SirenLayer


class Schrodinger(ABC):
    """
    Base class defining the interface for solving the time-dependent
    Schrödinger equation.
    """

    def __init__(
        self,
        coordinates: list[np.ndarray],
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
            coordinates: A list of numpy arrays representing the coordinate
                ranges.
            V: A function representing the potential energy as a function of the
                coordinates.
            initial_conditions: A list of InitialCondition objects defining the
                initial conditions for the quantum system.
            hbar: The reduced Planck constant (default is 1.0).
            m: The mass of the particle (default is 1.0).
            hidden_size: The number of neurons in each hidden layer of the
                neural network (default is 64).
            num_layers: The number of hidden layers in the neural network
                (default is 3).
            first_layer_omega_0: Frequency scale for the first SIREN layer.
            hidden_omega_0: Frequency scale for the remaining SIREN layers.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.V = V
        self.hbar = float(hbar)
        self.m = float(m)
        self.dimension = len(coordinates)
        self.initial_conditions = initial_conditions
        self.coordinate_arrays = [self._as_numpy_array(values) for values in coordinates]
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.first_layer_omega_0 = first_layer_omega_0
        self.hidden_omega_0 = hidden_omega_0

        self.inputs = self._create_model_inputs()
        self._device_inputs = self.inputs.to(self.device)
        self.input_normalizer = InputNormalizer(self.inputs)
        self.model = self._initialize_model()
        self.loss_history = self._empty_history()
        self.initial_probability_mass: Optional[float] = None
        self.trained = False

    def _initialize_model(self) -> nn.Sequential:
        """
        Initialize the neural network model for approximating the solution to
        the Schrödinger equation.

        Returns:
            An instance of nn.Sequential representing the neural network model.
        """
        layers: list[nn.Module] = [
            SirenLayer(
                self.dimension,
                self.hidden_size,
                is_first=True,
                omega_0=self.first_layer_omega_0,
            )
        ]

        for _ in range(self.num_layers):
            layers.append(
                SirenLayer(
                    self.hidden_size,
                    self.hidden_size,
                    is_first=False,
                    omega_0=self.hidden_omega_0,
                )
            )

        final_linear = nn.Linear(self.hidden_size, 2)
        with torch.no_grad():
            bound = (6 / self.hidden_size) ** 0.5 / self.hidden_omega_0
            final_linear.weight.uniform_(-bound, bound)

        layers.append(final_linear)
        return nn.Sequential(*layers).to(self.device)

    def _as_numpy_array(self, values) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy().astype(np.float32)
        return np.asarray(values, dtype=np.float32)

    def _to_device_tensor(self, X) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(device=self.device, dtype=self.inputs.dtype)
        return torch.tensor(X, dtype=self.inputs.dtype, device=self.device)

    def _forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self._to_device_tensor(X)
        X_normalized = self.input_normalizer.normalize(X)
        return self.model(X_normalized)

    def _sample_training_inputs(self, noise_std: float) -> torch.Tensor:
        X = self._device_inputs.detach().clone()
        if noise_std > 0:
            X = X + torch.randn_like(X) * noise_std
        return X.requires_grad_(True)

    def _empty_history(self) -> dict[str, list[float]]:
        return {
            "total_loss": [],
            "ic_loss": [],
            "norm_loss": [],
            "norm_loss_raw": [],
            "de_loss": [],
            "de_loss_raw": [],
            "anchor_loss": [],
            "anchor_loss_raw": [],
            "probability_mass_mean": [],
            "probability_mass_min": [],
            "probability_mass_max": [],
        }

    def _record_history(
        self,
        history: dict[str, list[float]],
        ic_loss: float,
        norm_loss: float,
        norm_loss_raw: float,
        de_loss: float,
        de_loss_raw: float,
        anchor_loss: float,
        anchor_loss_raw: float,
        probability_mass_mean: float,
        probability_mass_min: float,
        probability_mass_max: float,
        total_loss: float,
    ) -> None:
        history["ic_loss"].append(float(ic_loss))
        history["norm_loss"].append(float(norm_loss))
        history["norm_loss_raw"].append(float(norm_loss_raw))
        history["de_loss"].append(float(de_loss))
        history["de_loss_raw"].append(float(de_loss_raw))
        history["anchor_loss"].append(float(anchor_loss))
        history["anchor_loss_raw"].append(float(anchor_loss_raw))
        history["probability_mass_mean"].append(float(probability_mass_mean))
        history["probability_mass_min"].append(float(probability_mass_min))
        history["probability_mass_max"].append(float(probability_mass_max))
        history["total_loss"].append(float(total_loss))

    def _capture_training_state(self, optimizer: Optional[torch.optim.Optimizer] = None) -> dict[str, object]:
        """
        Snapshot the current model state and, if provided, the optimizer state.
        """
        checkpoint: dict[str, object] = {
            "model": {k: v.detach().clone() for k, v in self.model.state_dict().items()},
        }
        if optimizer is not None:
            checkpoint["optimizer"] = copy.deepcopy(optimizer.state_dict())
        return checkpoint

    def _restore_training_state(
        self,
        checkpoint: dict[str, object],
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """
        Restore a previously captured model state and optional optimizer state.
        """
        self.model.load_state_dict(checkpoint["model"])
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

    def _overwrite_latest_history_entry(
        self,
        history: dict[str, list[float]],
        metrics: dict[str, float],
    ) -> None:
        """
        Replace the latest history entry so the tracked history matches the
        restored best model state for the phase.
        """
        for key, value in metrics.items():
            history[key][-1] = float(value)

    def get_coordinate(self, coordinate: int) -> np.ndarray:
        """
        Get the coordinate values for the given coordinate index.

        Parameters:
            coordinate: The index of the coordinate to retrieve.
        Returns:
            A numpy array of shape (num_points,) representing the values of the
            specified coordinate.
        """
        return self.inputs[:, coordinate].detach().cpu().numpy()

    def get_solution(self) -> pd.DataFrame:
        """
        Get the solution of the Schrödinger equation as a pandas DataFrame.

        Returns:
            A DataFrame with columns for each coordinate, the real and imaginary
            parts of the wavefunction, and the probability density.
        """
        raise NotImplementedError("Solution retrieval must be implemented in subclass.")

    def probability_density(self, X: torch.Tensor) -> np.ndarray:
        """
        Calculate the probability density of the wavefunction at the given input
        points X.

        Parameters:
            X: A tensor of shape (num_points, dimension) representing the input
                points.
        Returns:
            A numpy array of shape (num_points,) representing the probability
            density at the input points.
        """
        if not self.trained:
            raise ValueError("Model must be trained before calculating probability density.")
        psi = self.psi(X)
        return psi[:, 0] ** 2 + psi[:, 1] ** 2

    def psi(self, X: torch.Tensor) -> np.ndarray:
        """
        Calculate the wavefunction at the given input points X.

        Parameters:
            X: A tensor of shape (num_points, dimension) representing the input
                points.
        Returns:
            A numpy array of shape (num_points, 2) where the first column is the
            real part of the wavefunction and the second column is the
            imaginary part of the wavefunction at the input points.
        """
        if not self.trained:
            raise ValueError("Model must be trained before calculating wavefunction.")
        with torch.no_grad():
            psi = self._forward(X)
        return psi.detach().cpu().numpy()

    @abstractmethod
    def _create_model_inputs(self) -> torch.Tensor:
        """
        Create the input tensor for the neural network based on the coordinate
        ranges.

        Returns:
            A tensor of shape (num_points, dimension) where each row is a
            coordinate point in the input space.
        """
        raise NotImplementedError("Model input creation must be implemented in subclass.")

    @abstractmethod
    def _dt(self, f: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate the time derivative of the function f.

        Parameters:
            f: A tensor representing the function values at the input points X.
            X: A tensor of shape (num_points, dimension) representing the input
                points.
        Returns:
            A tensor of shape (num_points,) representing the time derivative of
            f at the input points.
        """
        raise NotImplementedError("Time derivative must be implemented in subclass.")

    @abstractmethod
    def _laplacian(self, psi: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Laplacian of the wavefunction psi.

        Parameters:
            psi: A tensor of shape (num_points,) representing the wavefunction
                values at the input points X.
            X: A tensor of shape (num_points, dimension) representing the input
                points.
        Returns:
            A tensor of shape (num_points,) representing the Laplacian of psi at
            the input points.
        """
        raise NotImplementedError("Laplacian must be implemented in subclass.")

    @abstractmethod
    def _V(self, X: torch.Tensor) -> torch.Tensor:
        """
        Map the components of X to be used in the potential function V.

        Parameters:
            X: A tensor of shape (num_points, dimension) representing the input
                points.
        Returns:
            A tensor of shape (num_points,) representing the potential values at
            the input points.
        """
        raise NotImplementedError("Mapping of V must be implemented in subclass.")

    @abstractmethod
    def _probability_mass_by_time(self, probability_density: torch.Tensor) -> torch.Tensor:
        """
        Integrate probability density over space for each time slice.

        Parameters:
            probability_density: A tensor of shape (num_points,) representing the
                probability density over the full grid.
        Returns:
            A tensor containing the total probability mass at each time slice.
        """
        raise NotImplementedError("Normalization integration must be implemented in subclass.")

    def ic_loss(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss for the initial conditions.

        Parameters:
            X: A tensor of shape (num_points, dimension) representing the input
                points.
        Returns:
            A scalar tensor representing the total loss for the initial
            conditions.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        for condition in self.initial_conditions:
            total_loss = total_loss + condition.loss(self._forward, X)
        return total_loss

    def anchor_loss(self, X: torch.Tensor, anchors: list[InitialCondition]) -> torch.Tensor:
        """
        Calculate the loss for time-slice anchoring conditions.

        Parameters:
            X: A tensor of shape (num_points, dimension) representing the input
                points.
            anchors: A list of InitialCondition objects representing known
                exact-solution snapshots at intermediate times.
        Returns:
            A scalar tensor representing the total anchor loss.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        for anchor in anchors:
            total_loss = total_loss + anchor.loss(self._forward, X)
        return total_loss

    def normalization_loss(self) -> torch.Tensor:
        """
        Calculate a loss term that encourages conservation of total probability
        mass over space at each time slice.

        Returns:
            A scalar tensor representing the normalization loss.
        """
        return self._probability_mass_metrics()["norm_loss_raw"]

    def _probability_mass_metrics(self) -> dict[str, torch.Tensor]:
        """
        Calculate probability-mass summary statistics for the current model.
        """
        psi = self._forward(self._device_inputs)
        probability_density = psi[:, 0] ** 2 + psi[:, 1] ** 2
        probability_mass = self._probability_mass_by_time(probability_density)
        return {
            "norm_loss_raw": torch.mean((probability_mass - 1.0) ** 2),
            "probability_mass_mean": probability_mass.mean(),
            "probability_mass_min": probability_mass.min(),
            "probability_mass_max": probability_mass.max(),
        }

    def _initial_condition_probability_mass(self) -> Optional[float]:
        """
        Estimate the probability mass implied by the earliest-time value
        initial condition, if exactly one such condition is present.
        """
        initial_time = float(np.min(self.coordinate_arrays[0]))
        eligible_conditions = [
            condition
            for condition in self.initial_conditions
            if condition.coordinate == 0
            and not condition.is_derivative
            and np.isclose(float(condition.coordinate_value), initial_time)
        ]
        if len(eligible_conditions) != 1:
            return None

        condition = eligible_conditions[0]
        with torch.no_grad():
            X_0 = condition.get_initial_inputs(self._device_inputs)
            initial_values = condition.get_initial_values(X_0)
            probability_density = initial_values[:, 0] ** 2 + initial_values[:, 1] ** 2
            probability_mass = self._probability_mass_by_time(probability_density)
        return float(probability_mass.mean().detach().cpu())

    def _warn_if_initial_condition_not_normalized(self, tolerance: float = 1e-2) -> Optional[float]:
        """
        Warn when the earliest-time value initial condition is not normalized.
        """
        self.initial_probability_mass = self._initial_condition_probability_mass()
        if self.initial_probability_mass is None:
            return None
        if abs(self.initial_probability_mass - 1.0) > tolerance:
            initial_time = float(np.min(self.coordinate_arrays[0]))
            warnings.warn(
                (
                    "Initial-condition probability mass at "
                    f"t={initial_time:g} is {self.initial_probability_mass:.6f}, "
                    "expected 1.0. Normalize the initial wavefunction or "
                    "adjust normalization handling before training."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        return self.initial_probability_mass

    def de_loss(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss for the Schrödinger equation.

        Parameters:
            X: A tensor of shape (num_points, dimension) representing the input
                points.
        Returns:
            A scalar tensor representing the total loss for the Schrödinger
            equation.
        """
        psi = self._forward(X)
        psi_R = psi[:, 0]
        psi_I = psi[:, 1]
        real_part = (
            self.hbar * self._dt(psi_I, X)
            - (self.hbar ** 2 / (2 * self.m)) * self._laplacian(psi_R, X)
            + self._V(X) * psi_R
        )
        imag_part = (
            -self.hbar * self._dt(psi_R, X)
            - (self.hbar ** 2 / (2 * self.m)) * self._laplacian(psi_I, X)
            + self._V(X) * psi_I
        )
        return nn.MSELoss()(real_part, torch.zeros_like(real_part)) + nn.MSELoss()(
            imag_part,
            torch.zeros_like(imag_part),
        )

    def train(
        self,
        num_epochs: int = 1000,
        learning_rate: float = 1e-3,
        phase1_learning_rate: Optional[float] = None,
        phase2_learning_rate: Optional[float] = None,
        frac_epochs_ic_only: float = 0.4,
        frac_epochs_to_fully_weight_de: float = 0.65,
        de_weight: float = 0.7,
        ic_weight: float = 1.0,
        norm_weight: float = 0.1,
        phase1_ic_weight: Optional[float] = None,
        phase1_norm_weight: Optional[float] = None,
        phase2_ic_weight: Optional[float] = None,
        phase2_norm_weight: Optional[float] = None,
        anchors: Optional[list[InitialCondition]] = None,
        anchor_weight: float = 1.0,
        phase2_anchor_weight: Optional[float] = None,
        noise_std: float = 0.01,
        print_every: Optional[int] = 100,
    ) -> dict[str, list[float]]:
        """
        Train the neural network to approximate the solution to the Schrödinger
        equation.

        Parameters:
            num_epochs: The number of training epochs.
            learning_rate: The learning rate for the optimizer.
            phase1_learning_rate: Optional override for the phase-1 learning
                rate. Defaults to `learning_rate`.
            phase2_learning_rate: Optional override for the phase-2 learning
                rate. Defaults to `learning_rate`.
            frac_epochs_ic_only: The fraction of epochs to allocate to initial
                condition and normalization training only.
            frac_epochs_to_fully_weight_de: The fraction of epochs by which the
                differential-equation loss should reach its full weight.
            de_weight: The weight for the differential equation loss.
            ic_weight: The weight for the initial condition loss.
            norm_weight: The weight for the normalization loss.
            phase1_ic_weight: Optional override for the phase-1 initial-
                condition weight. Defaults to `ic_weight`.
            phase1_norm_weight: Optional override for the phase-1
                normalization weight. Defaults to `norm_weight`.
            phase2_ic_weight: Optional override for the phase-2 initial-
                condition weight. Defaults to `ic_weight`.
            phase2_norm_weight: Optional override for the phase-2
                normalization weight. Defaults to `norm_weight`.
            anchors: Optional list of InitialCondition objects representing
                known exact-solution snapshots at intermediate times. Used
                as additional loss terms during phase 2 only. Each anchor's
                individual weight attribute is respected, then the total is
                multiplied by anchor_weight. Defaults to None (no anchors).
            anchor_weight: The global weight multiplier for the aggregate
                anchor loss. Defaults to 1.0.
            phase2_anchor_weight: Optional override for the phase-2 anchor
                weight. Defaults to anchor_weight.
            noise_std: The standard deviation of the noise added to the inputs
                during training for better generalization.
            print_every: If provided, the training loss will be printed every
                `print_every` epochs.
        Returns:
            A dictionary containing the tracked loss history for each phase.
        """
        self.model.train()
        history = self._empty_history()
        self._warn_if_initial_condition_not_normalized()
        phase_1_learning_rate = learning_rate if phase1_learning_rate is None else phase1_learning_rate
        phase_2_learning_rate = learning_rate if phase2_learning_rate is None else phase2_learning_rate
        optimizer = torch.optim.Adam(self.model.parameters(), lr=phase_1_learning_rate)
        phase_1_ic_weight = ic_weight if phase1_ic_weight is None else phase1_ic_weight
        phase_1_norm_weight = norm_weight if phase1_norm_weight is None else phase1_norm_weight
        phase_2_ic_weight = ic_weight if phase2_ic_weight is None else phase2_ic_weight
        phase_2_norm_weight = norm_weight if phase2_norm_weight is None else phase2_norm_weight
        phase_2_anchor_weight = anchor_weight if phase2_anchor_weight is None else phase2_anchor_weight
        anchors = anchors or []
        phase_1_epochs = int(num_epochs * frac_epochs_ic_only)
        phase_2_epochs = max(num_epochs - phase_1_epochs, 0)
        de_ramp_epochs = max(int(num_epochs * frac_epochs_to_fully_weight_de) - phase_1_epochs, 0)

        self.train_phase_1(
            optimizer=optimizer,
            history=history,
            num_epochs=phase_1_epochs,
            ic_weight=phase_1_ic_weight,
            norm_weight=phase_1_norm_weight,
            noise_std=noise_std,
            print_every=print_every,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = phase_2_learning_rate
        self.train_phase_2(
            optimizer=optimizer,
            history=history,
            num_epochs=phase_2_epochs,
            de_weight=de_weight,
            ic_weight=phase_2_ic_weight,
            norm_weight=phase_2_norm_weight,
            num_epochs_to_fully_weight_de=de_ramp_epochs,
            anchors=anchors,
            anchor_weight=phase_2_anchor_weight,
            noise_std=noise_std,
            print_every=print_every,
        )

        self.loss_history = history
        self.trained = True
        self.model.eval()
        return history

    def train_phase_1(
        self,
        optimizer: torch.optim.Optimizer,
        history: dict[str, list[float]],
        num_epochs: int,
        ic_weight: float = 1.0,
        norm_weight: float = 1.0,
        noise_std: float = 0.01,
        print_every: Optional[int] = 100,
    ) -> None:
        """
        Train the model for the initial conditions and normalization only.

        Parameters:
            optimizer: The optimizer used to update model parameters.
            history: The dictionary used to collect loss history.
            num_epochs: The number of epochs to train in phase 1.
            ic_weight: The weight for the initial condition loss.
            norm_weight: The weight for the normalization loss.
            noise_std: The standard deviation of the noise added to the inputs
                during training.
            print_every: If provided, the training loss will be printed every
                `print_every` epochs.
        """
        best_loss = float("inf")
        best_epoch: Optional[int] = None
        best_checkpoint: Optional[dict[str, object]] = None
        best_metrics: Optional[dict[str, float]] = None
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            X = self._sample_training_inputs(noise_std)
            ic_loss = ic_weight * self.ic_loss(X)
            norm_metrics = self._probability_mass_metrics()
            norm_loss_raw = norm_metrics["norm_loss_raw"]
            norm_loss = norm_weight * norm_loss_raw
            loss = ic_loss + norm_loss
            if loss.item() < best_loss:
                best_loss = float(loss.item())
                best_epoch = epoch
                best_checkpoint = self._capture_training_state(optimizer)
                best_metrics = {
                    "ic_loss": ic_loss.item(),
                    "norm_loss": norm_loss.item(),
                    "norm_loss_raw": norm_loss_raw.item(),
                    "de_loss": 0.0,
                    "de_loss_raw": 0.0,
                    "anchor_loss": 0.0,
                    "anchor_loss_raw": 0.0,
                    "probability_mass_mean": norm_metrics["probability_mass_mean"].item(),
                    "probability_mass_min": norm_metrics["probability_mass_min"].item(),
                    "probability_mass_max": norm_metrics["probability_mass_max"].item(),
                    "total_loss": loss.item(),
                }
            loss.backward()
            optimizer.step()
            self._record_history(
                history,
                ic_loss.item(),
                norm_loss.item(),
                norm_loss_raw.item(),
                0.0,
                0.0,
                0.0,
                0.0,
                norm_metrics["probability_mass_mean"].item(),
                norm_metrics["probability_mass_min"].item(),
                norm_metrics["probability_mass_max"].item(),
                loss.item(),
            )
            if print_every and epoch % print_every == 0:
                print(
                    f"Phase 1 Epoch {epoch}: "
                    f"IC={ic_loss.item():.6f}, Norm={norm_loss.item():.6f} "
                    f"(raw={norm_loss_raw.item():.6f}), "
                    "Mass(mean/min/max)="
                    f"{norm_metrics['probability_mass_mean'].item():.6f}/"
                    f"{norm_metrics['probability_mass_min'].item():.6f}/"
                    f"{norm_metrics['probability_mass_max'].item():.6f}, "
                    f"Total={loss.item():.6f}"
                )
        if best_checkpoint is not None and best_metrics is not None:
            self._restore_training_state(best_checkpoint, optimizer)
            self._overwrite_latest_history_entry(history, best_metrics)
            if print_every and best_epoch is not None and best_epoch != num_epochs - 1:
                print(
                    f"Phase 1 restore best checkpoint from epoch {best_epoch}: "
                    f"Total={best_loss:.6f}"
                )

    def train_phase_2(
        self,
        optimizer: torch.optim.Optimizer,
        history: dict[str, list[float]],
        num_epochs: int,
        de_weight: float,
        ic_weight: float,
        norm_weight: float,
        num_epochs_to_fully_weight_de: int = 500,
        anchors: Optional[list[InitialCondition]] = None,
        anchor_weight: float = 0.0,
        noise_std: float = 0.01,
        print_every: Optional[int] = 100,
    ) -> None:
        """
        Train the model for both the initial conditions and the differential
        equation.

        Parameters:
            optimizer: The optimizer used to update model parameters.
            history: The dictionary used to collect loss history.
            num_epochs: The number of epochs to train in phase 2.
            de_weight: The weight for the differential equation loss.
            ic_weight: The weight for the initial condition loss.
            norm_weight: The weight for the normalization loss.
            num_epochs_to_fully_weight_de: The number of phase-2 epochs used to
                ramp the differential-equation loss to full weight.
            anchors: Optional list of InitialCondition objects representing
                known exact-solution snapshots at intermediate times.
            anchor_weight: The global weight multiplier for the aggregate
                anchor loss. Defaults to 0.0 (disabled when called directly
                without anchors).
            noise_std: The standard deviation of the noise added to the inputs
                during training.
            print_every: If provided, the training loss will be printed every
                `print_every` epochs.
        """
        anchors_list = anchors or []
        best_loss = float("inf")
        best_epoch: Optional[int] = None
        best_checkpoint: Optional[dict[str, object]] = None
        best_metrics: Optional[dict[str, float]] = None
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            X = self._sample_training_inputs(noise_std)
            de_weight_adjusted = min(
                de_weight,
                de_weight * (epoch + 1) / max(num_epochs_to_fully_weight_de, 1),
            )
            ic_loss = ic_weight * self.ic_loss(X)
            de_loss_raw = self.de_loss(X)
            de_loss = de_weight_adjusted * de_loss_raw
            norm_metrics = self._probability_mass_metrics()
            norm_loss_raw = norm_metrics["norm_loss_raw"]
            norm_loss = norm_weight * norm_loss_raw
            if anchors_list:
                anchor_loss_raw = self.anchor_loss(X, anchors_list)
                anchor_loss = anchor_weight * anchor_loss_raw
            else:
                anchor_loss_raw = torch.tensor(0.0, device=self.device)
                anchor_loss = torch.tensor(0.0, device=self.device)
            loss = ic_loss + de_loss + norm_loss + anchor_loss
            if loss.item() < best_loss:
                best_loss = float(loss.item())
                best_epoch = epoch
                best_checkpoint = self._capture_training_state(optimizer)
                best_metrics = {
                    "ic_loss": ic_loss.item(),
                    "norm_loss": norm_loss.item(),
                    "norm_loss_raw": norm_loss_raw.item(),
                    "de_loss": de_loss.item(),
                    "de_loss_raw": de_loss_raw.item(),
                    "anchor_loss": anchor_loss.item(),
                    "anchor_loss_raw": anchor_loss_raw.item(),
                    "probability_mass_mean": norm_metrics["probability_mass_mean"].item(),
                    "probability_mass_min": norm_metrics["probability_mass_min"].item(),
                    "probability_mass_max": norm_metrics["probability_mass_max"].item(),
                    "total_loss": loss.item(),
                }
            loss.backward()
            # Prevent DE-loss gradients from destroying IC-learned weights in a single step.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            self._record_history(
                history,
                ic_loss.item(),
                norm_loss.item(),
                norm_loss_raw.item(),
                de_loss.item(),
                de_loss_raw.item(),
                anchor_loss.item(),
                anchor_loss_raw.item(),
                norm_metrics["probability_mass_mean"].item(),
                norm_metrics["probability_mass_min"].item(),
                norm_metrics["probability_mass_max"].item(),
                loss.item(),
            )
            if print_every and epoch % print_every == 0:
                anchor_str = ""
                if anchors_list:
                    anchor_str = (
                        f"Anchor={anchor_loss.item():.6f} "
                        f"(raw={anchor_loss_raw.item():.6f}), "
                    )
                print(
                    f"Phase 2 Epoch {epoch}: "
                    f"IC={ic_loss.item():.6f}, DE={de_loss.item():.6f} "
                    f"(raw={de_loss_raw.item():.6f}), "
                    f"{anchor_str}"
                    f"Norm={norm_loss.item():.6f} "
                    f"(raw={norm_loss_raw.item():.6f}), "
                    "Mass(mean/min/max)="
                    f"{norm_metrics['probability_mass_mean'].item():.6f}/"
                    f"{norm_metrics['probability_mass_min'].item():.6f}/"
                    f"{norm_metrics['probability_mass_max'].item():.6f}, "
                    f"Total={loss.item():.6f}"
                )
        if best_checkpoint is not None and best_metrics is not None:
            self._restore_training_state(best_checkpoint, optimizer)
            self._overwrite_latest_history_entry(history, best_metrics)
            if print_every and best_epoch is not None and best_epoch != num_epochs - 1:
                print(
                    f"Phase 2 restore best checkpoint from epoch {best_epoch}: "
                    f"Total={best_loss:.6f}"
                )
