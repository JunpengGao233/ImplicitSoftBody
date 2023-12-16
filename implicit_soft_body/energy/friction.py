import torch
from torch import nn

from typing import Union

from .base import EnergyFunc
class FrictionEnergy(EnergyFunc):
    def __init__(
        self,
        k: Union[torch.Tensor, float],
        epsilon: Union[torch.Tensor, float],
        dt: Union[torch.Tensor, float],
    ):
        """
        Args:
            k: Friction energy coefficient
            epsilon: Friction epsilon
        """
        self.__epsilon: Union[torch.Tensor, float] = epsilon
        self.__k: Union[torch.Tensor, float] = k
        self.__dt: Union[torch.Tensor, float] = dt

    def forward(self, x0: torch.Tensor, x:torch.Tensor) -> torch.Tensor:
        vx = (x - x0) / self.__dt
        return (
            0.5
            * self.__k
            * torch.sum(
                torch.sum(vx[:, :-1] ** 2, axis=-1) * nn.functional.relu(-x0[:, -1] + self.__epsilon)
            )
        )
