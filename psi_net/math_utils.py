"""
Module for mathematical utilities used in the quantum system simulations.
"""

import torch

def diff(f: torch.Tensor, x: torch.Tensor, coordinate: int, order=1):
    """
    Calculate the nth order derivative of a function f with respect to x using PyTorch's autograd.

    Parameters:
        f: A tensor representing the function values at the input points x.
        x: A tensor of shape (num_points, dimension) representing the input points.
        coordinate: The index of the coordinate with respect to which the derivative is taken.
        order: The order of the derivative to calculate (default is 1).
    Returns:
        A tensor of shape (num_points,) representing the nth order derivative of f with respect to the specified coordinate at the input points.
    """
    if order < 0:
        raise ValueError("Order of derivative must be non-negative.")
    if order == 0:
        return f
    elif order == 1:
        return torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True)[0][:, coordinate]
    return diff(diff(f, x, coordinate, order - 1), x, coordinate, 1)

def heaviside(x):
    """
    Heaviside step function.
    """
    return 0.5 * (torch.sign(x) + 1)

def square_wave(x, center, width):
    """
    Square wave function centered at 'center' with given 'width'.
    """
    return heaviside(x - (center - width/2)) - heaviside(x - (center + width/2))