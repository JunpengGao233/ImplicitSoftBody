import jax
import jax.numpy as jnp
from typing import Callable, Union
from .base import EnergyFunc
from ..geometry.triangle import Triangle


class TriangleEnergy(EnergyFunc):
    def __init__(self, mu: float, lamb: float):
        """
        Args:
            mu: Neo-Hookean energy coefficient
            lamb: Neo-Hookean energy coefficient

        """
        self.__mu = mu
        self.__lamb = lamb

    def forward(
        self, x0: jax.Array, x: jax.Array
    ) -> jax.Array:
        """
        Forward Neo-Hookean energy function.

        Args:
            x0: (N, 3, dim) array of undeformed triangle vertices
            x: (N, 3, dim) array of deformed triangle vertices

        """
        verts1 = x0
        verts2 = x
        mu = self.__mu
        lamb = self.__lamb
        dim = verts1.shape[-1]
        F = jnp.zeros((verts1.shape[0], dim, dim))
        for i in range(3):
            # Compute the basis vectors of the original triangle
            e = verts1[(i + 1) % 3] - verts1[i]
            f = verts1[(i + 2) % 3] - verts1[i]

            # Compute the basis vectors of the deformed triangle
            e_d = verts2[(i + 1) % 3] - verts2[i]
            f_d = verts2[(i + 2) % 3] - verts2[i]

            # Construct the deformation gradient F
            F += jnp.outer(e_d, e) + jnp.outer(f_d, f)

        # Calculate the Jacobian of the deformation
        J = jnp.linalg.det(F)
        I = jnp.dot(F.T, F)
        E = (
            (mu / 2) * (jnp.trace(I) - 2)
            - mu * jnp.log(J)
            + (lamb / 2) * jnp.log(J) ** 2
        )

        E = jnp.sum(E)

        return E

