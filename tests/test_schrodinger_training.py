import math
import warnings

import pytest
import torch

from psi_net.cartesian.cartesian_schrodinger import CartesianSchrodinger
from psi_net.cartesian.coordinates import CartesianCoordinates
from psi_net.cartesian.initial_conditions import CartesianInitialCondition


def make_solver(
    *,
    normalized: bool = False,
    time_points: int = 2,
    space_points: int = 3,
) -> CartesianSchrodinger:
    t = torch.linspace(0.0, 1.0, time_points)
    x = torch.linspace(0.0, 1.0, space_points)
    amplitude = math.sqrt(2.0) if normalized else 1.0

    def psi_r_initial(_t, position):
        return amplitude * torch.sin(torch.pi * position)

    def psi_i_initial(_t, position):
        return torch.zeros_like(position)

    def zero_boundary(time_value, _position):
        return torch.zeros_like(time_value)

    return CartesianSchrodinger(
        coordinates=[t, x],
        V=lambda _t, position: torch.zeros_like(position),
        initial_conditions=[
            CartesianInitialCondition(CartesianCoordinates.T, 0.0, psi_r_initial, psi_i_initial),
            CartesianInitialCondition(CartesianCoordinates.X, 0.0, zero_boundary, zero_boundary),
            CartesianInitialCondition(CartesianCoordinates.X, 1.0, zero_boundary, zero_boundary),
        ],
        hidden_size=8,
        num_layers=1,
    )


def test_exact_ground_state_has_near_zero_de_and_normalization_loss():
    solver = make_solver(normalized=True, time_points=5, space_points=101)
    energy = math.pi**2 / 2.0

    def exact_forward(X):
        time_value = X[:, CartesianCoordinates.T]
        position = X[:, CartesianCoordinates.X]
        amplitude = math.sqrt(2.0) * torch.sin(torch.pi * position)
        real = amplitude * torch.cos(energy * time_value)
        imag = -amplitude * torch.sin(energy * time_value)
        return torch.stack((real, imag), dim=1)

    solver._forward = exact_forward
    X = solver._sample_training_inputs(noise_std=0.0)

    assert solver.ic_loss(X).item() == pytest.approx(0.0, abs=1e-6)
    assert solver.normalization_loss().item() == pytest.approx(0.0, abs=1e-6)
    assert solver.de_loss(X).item() == pytest.approx(0.0, abs=1e-6)


def test_warns_for_unnormalized_time_initial_condition():
    solver = make_solver(normalized=False, time_points=5, space_points=101)

    with pytest.warns(RuntimeWarning, match="0.500000"):
        initial_mass = solver._warn_if_initial_condition_not_normalized()

    assert initial_mass == pytest.approx(0.5, abs=1e-6)


def test_does_not_warn_for_normalized_time_initial_condition():
    solver = make_solver(normalized=True, time_points=5, space_points=101)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        initial_mass = solver._warn_if_initial_condition_not_normalized()

    assert not recorded
    assert initial_mass == pytest.approx(1.0, abs=1e-6)


def test_anchor_loss_method_sums_individual_losses():
    solver = make_solver(normalized=True, time_points=3, space_points=5)
    anchor1 = CartesianInitialCondition(
        CartesianCoordinates.T, 0.5,
        psi_R=lambda t, x: torch.zeros_like(x),
        psi_I=lambda t, x: torch.zeros_like(x),
        weight=1.0,
    )
    anchor2 = CartesianInitialCondition(
        CartesianCoordinates.T, 0.5,
        psi_R=lambda t, x: torch.zeros_like(x),
        psi_I=lambda t, x: torch.zeros_like(x),
        weight=2.0,
    )
    X = solver._sample_training_inputs(noise_std=0.0)
    loss_1 = anchor1.loss(solver._forward, X)
    loss_2 = anchor2.loss(solver._forward, X)
    combined = solver.anchor_loss(X, [anchor1, anchor2])
    assert combined.item() == pytest.approx((loss_1 + loss_2).item(), abs=1e-6)
