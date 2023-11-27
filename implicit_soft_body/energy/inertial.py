from typing import Union

import torch

from .base import EnergyFunc


class InertialEnergy(EnergyFunc):
    def __init__(
        self,
        m: torch.Tensor,
        h: Union[float, torch.Tensor] = 0.01,
    ):
        """
        Args:
            m: (n,) array of masses or float if all masses are the same
            h: (1,) array of time step
        """
        self.__m = m
        self.__h = h

    def forward(self, x: torch.Tensor, x0: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        """
        Forward inertial energy function.

        Args:
            x: (n, 3) array of positions or (n, 2). (x,y,z) or (x,y)
        Returns:
            Inertial energy.
        """
        estimate_pos = x0 + self.__h * v0
        return 0.5 * self.__m * torch.sum((x - estimate_pos) ** 2)
