import jax
import jax.numpy as jnp
from typing import Callable, Union
from .base import EnergyFunc

class ElementEnergy(EnergyFunc):
    def __init__(self,  *args, **kwargs):
        # TODO: The Element Energy Function is not implemented yet.
        element_energy_func = lambda x0, vx0,x, action: 0.0
        super().__init__(element_energy_func, *args, **kwargs)

    def backward(self, *args, **kwargs):
        return None