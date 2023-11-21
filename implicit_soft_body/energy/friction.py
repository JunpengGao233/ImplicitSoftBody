import jax.numpy as jnp
import jax
import jax.nn as nn
from typing import Union
from .base import EnergyFunc


def friction_energy(
    x0: jax.Array,
    x: jax.Array,
    k: Union[jax.Array, float],
    epsilon: Union[jax.Array, float],
    dt: Union[jax.Array, float],
) -> jax.Array:
    """
    Friction energy function.

    Args:
        x: (n, 2) or (n,3) array of positions. (x,y) or (x,y,z)
        vx: (n, 2) or (n,3) array of velocities. (vx, vy) or (vx, vy, vz)
        k: Friction energy coefficient
        epsilon: Friction epsilon

    Returns:
        Friction energy.
    """
    # Compute energy.
    vx = (x - x0) / dt
    return (
        0.5
        * k
        * jnp.sum(jnp.sum(vx[:, :-1] ** 2, axis=-1) * nn.relu(-x[:, -1] + epsilon))
    )


# TODO: didn't consider time-variant case


class FrictionEnergy(EnergyFunc):
    def __init__(
        self,
        k: Union[jax.Array, float],
        epsilon: Union[jax.Array, float],
        dt: Union[jax.Array, float],
        *args,
        **kwargs,
    ):
        """
        Args:
            k: Friction energy coefficient
            epsilon: Friction epsilon
        """
        self.__epsilon: Union[jax.Array, float] = epsilon
        self.__k: Union[jax.Array, float] = k
        self.__dt: Union[jax.Array, float] = dt
        friction_energy_func = lambda x0, vx0, x, action: friction_energy(
            x0, x, k, epsilon, dt
        )
        super().__init__(friction_energy_func, *args, **kwargs)

    def backward(self, *args, **kwargs):
        return None  # Not implemented yet

    @property
    def k(self) -> Union[jax.Array, float]:
        return self.__k

    @property
    def epsilon(self) -> Union[jax.Array, float]:
        return self.__epsilon

    @property
    def dt(self) -> Union[jax.Array, float]:
        return self.__dt


if __name__ == "__main__":
    sample_x0 = jnp.zeros((10, 3))
    sample_vx0 = jnp.zeros((10, 3))
    sample_x = jnp.zeros((10, 3))
    sample_action = jnp.zeros((10, 3))

    energy = friction_energy(sample_x0, sample_x, 20000, 0.0, 0.1)
    energy_func = FrictionEnergy(20000, 0.0, 0.1)
    energy_func.grad(sample_x0, sample_vx0, sample_x, sample_action)
    print("Everything passed", energy)
