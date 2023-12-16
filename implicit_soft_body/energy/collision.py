import torch
from torch import nn
from typing import Union

from .base import EnergyFunc


class CollisionEnergy(EnergyFunc):
    def __init__(
        self,
        k: Union[torch.Tensor, float],
    ):
        """
        Args:
            k: Collision energy coefficient
        """
        self.__k: Union[torch.Tensor, float] = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Collision energy function.

        Args:
            x: (n, 3) array of positions or (n, 2). (x,y,z) or (x,y)
        Returns:
            Collision energy.
        """

        return self.__k * torch.sum(nn.functional.relu(-x[:, -1]) ** 2)
