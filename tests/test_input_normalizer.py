import torch

from psi_net.abstract.input_normalizer import InputNormalizer


def test_normalizer_handles_constant_dimensions():
    inputs = torch.tensor([[1.0, 2.0], [1.0, 4.0]], dtype=torch.float32)
    normalizer = InputNormalizer(inputs)
    normalized = normalizer.normalize(inputs)

    assert torch.isfinite(normalized).all()
    assert torch.allclose(normalizer.denormalize(normalized), inputs)
