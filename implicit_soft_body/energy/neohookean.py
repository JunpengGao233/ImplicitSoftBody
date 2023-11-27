import torch
from typing import Callable, Union
from .base import EnergyFunc
# from ..geometry.triangle import Triangle


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
        self, x0: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
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
        F = torch.zeros((verts1.shape[0], dim, dim))
        for i in range(3):
            # Compute the basis vectors of the original triangle
            e = verts1[:, (i + 1) % 3] - verts1[:,i]
            f = verts1[:, (i + 2) % 3] - verts1[:, i]

            # Compute the basis vectors of the deformed triangle
            e_d = verts2[:, (i + 1) % 3] - verts2[:, i]
            f_d = verts2[:, (i + 2) % 3] - verts2[:, i]

            # Construct the deformation gradient F
            F += torch.outer(e_d, e) + torch.outer(f_d, f)

        # Calculate the Jacobian of the deformation
        J = torch.det(F)
        I = torch.matmul(F.transpose(0, 2, 1), F)
        E = (
            (mu / 2) * (torch.trace(I) - 2)
            - mu * torch.log(J)
            + (lamb / 2) * torch.log(J) ** 2
        )

        E = torch.sum(E)

        return E

