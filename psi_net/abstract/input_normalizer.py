"""
Module for normalizing the input to the neural network. This is important for
training stability and convergence.
"""

from __future__ import annotations

import torch


class InputNormalizer:
    """
    Class for normalizing the input to the neural network. This is important for
    training stability and convergence.
    """

    def __init__(self, inputs: torch.Tensor, normalized_min: float = -1.0, normalized_max: float = 1.0):
        """
        Initialize the InputNormalizer.

        Parameters
        ----------
        inputs : torch.Tensor
            The inputs to be normalized. This can be any number of inputs, and
            they will be normalized independently.
        """
        self.inputs = inputs
        self.normalized_min = float(normalized_min)
        self.normalized_max = float(normalized_max)
        self.min_values = torch.min(inputs, dim=0).values
        self.max_values = torch.max(inputs, dim=0).values
        raw_span = self.max_values - self.min_values
        self.span = torch.where(raw_span == 0, torch.ones_like(raw_span), raw_span)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input data.

        Parameters
        ----------
        x : torch.Tensor
            The input data to be normalized.

        Returns
        -------
        torch.Tensor
            The normalized input data.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=self.min_values.dtype, device=self.min_values.device)
        min_values = self.min_values.to(device=x.device, dtype=x.dtype)
        span = self.span.to(device=x.device, dtype=x.dtype)
        scale = (self.normalized_max - self.normalized_min) / span
        return self.normalized_min + (x - min_values) * scale

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the input data.

        Parameters
        ----------
        x : torch.Tensor
            The normalized input data to be denormalized.

        Returns
        -------
        torch.Tensor
            The denormalized input data.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=self.min_values.dtype, device=self.min_values.device)
        min_values = self.min_values.to(device=x.device, dtype=x.dtype)
        span = self.span.to(device=x.device, dtype=x.dtype)
        scale = span / (self.normalized_max - self.normalized_min)
        return min_values + (x - self.normalized_min) * scale
