import jax.numpy as jnp
import jax
import jax.nn as nn
from typing import Union


class FrictionEnergy(EnergyFunc):
    def __init__(
        self,
        k: Union[jax.Array, float],
        epsilon: Union[jax.Array, float],
        dt: Union[jax.Array, float],
    ):
        """
        Args:
            k: Friction energy coefficient
            epsilon: Friction epsilon
        """
        self.__epsilon: Union[jax.Array, float] = epsilon
        self.__k: Union[jax.Array, float] = k
        self.__dt: Union[jax.Array, float] = dt

    def forward(self, x0: jax.Array, x:jax.Array) -> jax.Array:
        vx = (x - x0) / self.__dt
        return (
            0.5
            * self.__k
            * jnp.sum(
                jnp.sum(vx[:, :-1] ** 2, axis=-1) * nn.relu(-x[:, -1] + self.__epsilon)
            )
        )
