from typing import Union

import torch

from .base import EnergyFunc


class SpringEnergy(EnergyFunc):
    def __init__(
        self,
        l0: torch.Tensor,
        k: Union[torch.Tensor, float],
    ):
        self.__l0: torch.Tensor = l0
        self.__k: Union[torch.Tensor, float] = k

    def forward(self, x0: torch.Tensor, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Forward spring energy function.

        Args:
            x0: (n, dim) array of one end of spring nodes
            x: (n, dim) array of the other end of spring nodes
            a: (n, ) array of the actuation of the spring

        """
        l = torch.norm(x0 - x, dim=-1)
        return 0.5 * self.__k * torch.sum((l / (self.__l0 * a) - 1) ** 2)
