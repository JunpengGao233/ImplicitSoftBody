import torch
from torch import nn
from typing import Union
from .base import EnergyFunc


class GravityEnergy(EnergyFunc):
    def __init__(self, m: torch.Tensor, g: Union[float, torch.Tensor] = 9.8):
        """
        Args:
            m: (n,) array of masses
            g: (1,) array of gravity acceleration
        """
        self.__g = g
        self.__m = m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward gravity energy function.

        Args:
            x: (n, 2) array of positions or (n, 2). (x,y,z) or (x,y)
            m: (n,) array of masses or float if all masses are the same
            g: (1,) array of gravity acceleration
        Returns:
            Gravity energy.
        """
        return torch.sum(self.__m * self.__g * x[:, -1])
