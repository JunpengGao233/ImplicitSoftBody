from typing import Union

import torch

from .base import EnergyFunc


class InertialEnergy(EnergyFunc):
    def __init__(
        self,
        m: Union[torch.Tensor, float],
        h: float = 0.01,
        device:Union[torch.device, str]="cpu"
    ):
        """
        Args:
            m: (n,) array of masses or float if all masses are the same it can be float
            h: (float), timestep
        """
        if isinstance(m, torch.Tensor):
            self.__m = m.to(device)
        else:
            self.__m = m

        self.__h = h
        self.device = device

    def forward(self, x: torch.Tensor, x0: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
        """
        Forward inertial energy function.

        Args:
            x: (n, 3) array of positions or (n, 2). (x,y,z) or (x,y)
        Returns:
            Inertial energy.
        """
        estimate_pos = x0 + self.__h * v0
        return self.__m * torch.sum((x - estimate_pos) ** 2)
