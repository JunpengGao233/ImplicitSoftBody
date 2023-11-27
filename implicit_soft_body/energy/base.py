from abc import abstractmethod, ABC
import jax
from jax import grad
from typing import Any, Callable


class EnergyFunc(ABC):
    def forward(
        self,
        *args,
        **kwargs,
    ) -> jax.Array:
        """
        Compute energy with given position, velocity and action.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
