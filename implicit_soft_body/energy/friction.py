import torch
from torch import nn

from typing import Union

from .base import EnergyFunc
class FrictionEnergy(EnergyFunc):
    def __init__(
        self,
        k: Union[torch.Tensor, float],
        epsilon: Union[torch.Tensor, float],
        dt: float,
        device: Union[torch.Tensor, float] = 'cpu'
    ):
        """
        Args:
            k: Friction energy coefficient
            epsilon: Friction epsilon
            dt: timpstep (or called h)
        """
        if isinstance(epsilon, torch.Tensor):
            self.__epsilon =  epsilon.to(device)
        else:
            self.__epsilon = epsilon
        
        if isinstance(k, torch.Tensor):
            self.__k = k.to(device)
        else:
            self.__k = k
        
        self.__dt:float = dt
        self.device = device

    def forward(self, x0: torch.Tensor, x:torch.Tensor) -> torch.Tensor:
        vx = (x[:, 0] - x0[:, 0]) / self.__dt
        friction_energy = self.__k * vx * vx * torch.nn.functional.relu(-(x0[:,1]-self.__epsilon))
        return friction_energy.sum()
