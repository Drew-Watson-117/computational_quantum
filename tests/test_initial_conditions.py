import torch

from psi_net.abstract.initial_conditions import InitialCondition


def test_derivative_initial_condition_is_component_wise():
    condition = InitialCondition(
        coordinate=0,
        coordinate_value=0.0,
        psi_R=lambda t, x: 2.0 * t,
        psi_I=lambda t, x: torch.full_like(t, 3.0),
        derivative=True,
    )
    X = torch.tensor([[0.5, 1.0], [1.0, 2.0]], dtype=torch.float32, requires_grad=True)

    def model_fn(values):
        return torch.stack((values[:, 0] ** 2 + values[:, 1], 3.0 * values[:, 0] - values[:, 1]), dim=1)

    loss = condition.loss(model_fn, X)

    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
