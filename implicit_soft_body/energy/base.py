from abc import ABC

import torch


class EnergyFunc(ABC):
    def forward(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute energy with given position, velocity and action.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
