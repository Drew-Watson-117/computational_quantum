"""
Module containing the Enum class for the Cartesian coordinate system used in CartesianSchrodinger.
"""

from enum import IntEnum

class CartesianCoordinates(IntEnum):
    """
    Enumeration for the indices of the coordinates in CartesianSchrodinger.
    """

    T = 0
    X = 1
    Y = 2
    Z = 3
