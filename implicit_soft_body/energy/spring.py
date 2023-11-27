from typing import Union

import jax
import jax.numpy as jnp

from .base import EnergyFunc


class SpringEnergy(EnergyFunc):
    def __init__(
        self,
        l0: jax.Array,
        k: Union[jax.Array, float],
    ):
        self.__l0: jax.Array = l0
        self.__k: Union[jax.Array, float] = k

    def forward(self, x0: jax.Array, x: jax.Array, a: jax.Array) -> jax.Array:
        """
        Forward spring energy function.

        Args:
            x0: (n, dim) array of one end of spring nodes
            x: (n, dim) array of the other end of spring nodes
            a: (n, ) array of the actuation of the spring

        """
        l = jnp.linalg.norm(x0 - x, axis=-1)
        return 0.5 * self.__k * jnp.sum((l / (self.__l0 * a) - 1) ** 2)
