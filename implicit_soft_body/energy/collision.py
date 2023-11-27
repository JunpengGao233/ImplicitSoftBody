import jax
import jax.numpy as jnp
import jax.nn as nn
from typing import Union


class CollisionEnergy(EnergyFunc):
    def __init__(
        self,
        k: Union[jax.Array, float],
    ):
        """
        Args:
            k: Collision energy coefficient
        """
        self.__k: Union[jax.Array, float] = k

    def forward(self, x: jax.Array) -> jax.Array:
        """
        Collision energy function.

        Args:
            x: (n, 3) array of positions or (n, 2). (x,y,z) or (x,y)
        Returns:
            Collision energy.
        """

        return 0.5 * self.__k * jnp.sum(nn.relu(-x[:, -1]) ** 2)
