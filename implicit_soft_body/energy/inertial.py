from typing import Union

import jax
import jax.numpy as jnp

from .base import EnergyFunc


class InertialEnergy(EnergyFunc):
    def __init__(
        self,
        m: jax.Array,
        h: Union[float, jax.Array] = 0.01,
    ):
        """
        Args:
            m: (n,) array of masses or float if all masses are the same
            h: (1,) array of time step
        """
        self.__m = m
        self.__h = h

    def forward(self, x: jax.Array, x0: jax.Array, v0: jax.Array) -> jax.Array:
        """
        Forward inertial energy function.

        Args:
            x: (n, 3) array of positions or (n, 2). (x,y,z) or (x,y)
        Returns:
            Inertial energy.
        """
        estimate_pos = x0 + self.__h * v0
        return 0.5 * self.__m * jnp.sum((x - estimate_pos) ** 2)
