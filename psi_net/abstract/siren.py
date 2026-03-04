"""
Module defining the SIREN layer for PINNs, which uses sine activation functions 
and specific weight initialization to enable better learning of high-frequency functions.
"""
from torch import nn
import torch

class SirenLayer(nn.Module):
    """
    SIREN layer for Physics-Informed Neural Networks (PINNs).
    """
    def __init__(self, in_features, out_features, 
                 is_first=False, omega_0=30.0):
        super().__init__()

        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0

        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the SIREN layer according to the SIREN paper.
        """
        with torch.no_grad():
            if self.is_first:
                # First layer initialization
                bound = 1 / self.in_features
            else:
                # Hidden layer initialization
                bound = (6 / self.in_features) ** 0.5 / self.omega_0

            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        """Forward pass through the SIREN layer."""
        return torch.sin(self.omega_0 * self.linear(x))
