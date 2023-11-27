import jax
import jax.numpy as jnp
import jax.nn as nn
from typing import Union
from .base import EnergyFunc


class GravityEnergy(EnergyFunc):
    def __init__(self, m: jax.Array, g: Union[float, jax.Array] = 9.8, *args, **kwargs):
        """
        Args:
            m: (n,) array of masses
            g: (1,) array of gravity acceleration
        """
        self.__g = g
        self.__m = m

    def forward(self, x: jax.Array) -> jax.Array:
        """
        Forward gravity energy function.

        Args:
            x: (n, 2) array of positions or (n, 2). (x,y,z) or (x,y)
            m: (n,) array of masses or float if all masses are the same
            g: (1,) array of gravity acceleration
        Returns:
            Gravity energy.
        """
        return jnp.sum(self.__m * self.__g * x[:, -1])
