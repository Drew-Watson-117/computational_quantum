"""
Module for normalizing the input to the neural network. This is important for training stability and convergence.
"""

import torch

class InputNormalizer:
    """
    Class for normalizing the input to the neural network. This is important for training stability and convergence.
    """

    def __init__(self, inputs: torch.Tensor, normalized_min = -1.0, normalized_max = 1.0):
        """
        Initialize the InputNormalizer.

        Parameters
        ----------
        inputs : torch.Tensor
            The inputs to be normalized. This can be any number of inputs, and they will be normalized independently.
        """
        self.inputs = inputs
        self.normalized_min = normalized_min
        self.normalized_max = normalized_max
        self.min_values = torch.min(inputs, dim=0).values
        self.max_values = torch.max(inputs, dim=0).values

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
        return self.normalized_min + (x - self.min_values) * (self.normalized_max - self.normalized_min) / (self.max_values - self.min_values)

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
        return self.min_values + (x - self.normalized_min) * (self.max_values - self.min_values) / (self.normalized_max - self.normalized_min)
