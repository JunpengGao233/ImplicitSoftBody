import torch
from typing import Callable, Union
from .base import EnergyFunc
# from ..geometry.triangle import Triangle


class TriangleEnergy(EnergyFunc):
    def __init__(self, mu: float, lamb: float, init_elements: torch.Tensor):
        """
        Args:
            mu: Neo-Hookean energy coefficient
            lamb: Neo-Hookean energy coefficient

        """
        self.__mu = mu
        self.__lamb = lamb
        num_triangles = init_elements.shape[0]
        weight_matrix = torch.zeros((num_triangles,2,2))
        e = init_elements[:, 1] - init_elements[:, 0]
        f = init_elements[:, 2] - init_elements[:, 0]
        # print(init_elements[0])
        weight_matrix[:,0] = e
        weight_matrix[:,1] = f
        weight_matrix = torch.transpose(weight_matrix, 1, 2)
        weight_matrix_inv = torch.inverse(weight_matrix)
        self.grad_undeformed_sample_weight = torch.zeros((num_triangles,3,2))
        self.grad_undeformed_sample_weight[:,0] = -weight_matrix_inv[:,0] - weight_matrix_inv[:,1]
        self.grad_undeformed_sample_weight[:,1] = weight_matrix_inv[:,0]
        self.grad_undeformed_sample_weight[:,2] = weight_matrix_inv[:,1]
        self.weight_matrix_inv = weight_matrix_inv
        # print(weight_matrix_inv[0])

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
        element_matrix = torch.zeros((x.shape[0],2,2))
        e = x[:, 1] - x[:, 0]
        f = x[:, 2] - x[:, 0]
        element_matrix[:,0] = e
        element_matrix[:,1] = f
        element_matrix = torch.transpose(element_matrix, 1, 2)
        
        # Calculate the Jacobian of the deformation
        F = torch.matmul(element_matrix, self.weight_matrix_inv)
        J = torch.det(F)
        I = torch.matmul(torch.transpose(F, 1, 2), F)
        #compute trace of I along the last two dims
        # print(I)
        trace_I = torch.diagonal(I, dim1=-2, dim2=-1).sum(-1)
        # trace_I = torch.einsum("ii->i", I)
        # print(trace_I)
        qlogJ = -1.5 + 2 * J - 0.5 * J * J
        psi_mu = 0.5 * mu * (trace_I - 2) - mu * qlogJ
        psi_lambda = 0.5 * lamb * qlogJ * qlogJ
        E = psi_mu + psi_lambda
        # print(J)
        # E = (
        #     (mu / 2) * (trace_I - 2)
        #     - mu * torch.log(J)
        #     + (lamb / 2) * torch.log(J) ** 2
        # )

        E = torch.sum(E)

        return E

