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
        for i in range(3):
            e = init_elements[:, 1] - init_elements[:, 0]
            f = init_elements[:, 2] - init_elements[:, 0]
        print(init_elements[0])
        weight_matrix[:,0] = e
        weight_matrix[:,1] = f
        print(weight_matrix[0])
        weight_matrix_inv = torch.inverse(weight_matrix)
        self.grad_undeformed_sample_weight = torch.zeros((num_triangles,3,2))
        self.grad_undeformed_sample_weight[:,0] = -weight_matrix_inv[:,0] - weight_matrix_inv[:,1]
        self.grad_undeformed_sample_weight[:,1] = weight_matrix_inv[:,0]
        self.grad_undeformed_sample_weight[:,2] = weight_matrix_inv[:,1]
        # print(weight_matrix_inv)
        # self.grad_undeformed_sample_weight = torch.tensor(
        #     [[[-2.973978042602539,3.2218103408813477],[0.24783125519752502,-5.824039459228516]],[[-4.945597171783447,-0.1978236883878708],[2.3738865852355957,-2.5717110633850098]],[[3.839442491531372,-3.0541014671325684],[-5.2356038093566895,1.1343804597854614]],[[3.9643208980560303,-0.8589359521865845],[-1.7178723812103271,-1.8500168323516846]],[[-3.463204860687256,2.754821538925171],[-1.7316027879714966,-1.6528923511505127]],[[-2.155172109603882,1.580459475517273],[-2.729886054992676,-2.44252872467041]],[[-0.1952170431613922,4.587604522705078],[-2.7330410480499268,-2.4402153491973877]],[[-1.5962440967559814,1.690140724182129],[-2.816901683807373,-0.9389669895172119]],[[1.9628455638885498,-1.7525408267974854],[-3.2947769165039062,0.5608130693435669]],[[2.4398996829986572,-2.583423137664795],[-4.449228763580322,0.7893791794776917]],[[0.37062767148017883,2.0847811698913574],[-3.2429919242858887,-1.575168251991272]],[[-3.1052870750427246,-1.6496847867965698],[3.008246898651123,-0.48520150780677795]],[[0.5385037064552307,2.3694143295288086],[-3.984924554824829,-4.200326442718506]],[[3.004044771194458,-0.11554037034511566],[-2.618910551071167,1.810129165649414]],[[3.049201726913452,-0.10395022481679916],[-0.2772001326084137,1.5246014595031738]],[[-3.0250132083892822,0.11634685099124908],[3.1025776863098145,1.5900715589523315]],[[0.08276424556970596,1.9035797119140625],[-2.8967514038085938,0.04138179495930672]],[[-4.4989800453186035,-4.294477939605713],[-0.8179954886436462,2.2494893074035645]],[[-2.8865976333618164,0.2749159336090088],[3.024054527282715,2.8865954875946045]],[[-1.2121210098266602,-2.2510826587677],[-2.4242420196533203,5.0216450691223145]],[[5.385330677032471,-0.9285058379173279],[-3.899722099304199,2.971216917037964]],[[3.7119526863098145,-0.14847679436206818],[-0.14847798645496368,2.6726059913635254]],[[0.9840090870857239,-2.706027030944824],[5.16605281829834,2.460026502609253]],[[3.4567861557006836,-0.3292199671268463],[-7.407398223876953,-2.4691314697265625]],[[-4.081632614135742,6.802722454071045],[3.8548755645751953,-2.7210896015167236]],[[1.2802923917770386,2.3776867389678955],[3.1092820167541504,-3.7494282722473145]],[[-0.8274232149124146,5.082742691040039],[-1.300236463546753,-1.5366426706314087]],[[-3.3175339698791504,-1.579779028892517],[-2.685622453689575,1.8957343101501465]],[[0,4.761904716491699],[-1.6666666269302368,-1.3095234632492065]],[[-3.118907928466797,0.38986310362815857],[0.8447044491767883,-1.4944767951965332]],[[3.1582565307617188,1.4074833393096924],[-0.13731537759304047,-1.5104701519012451]],[[-2.433863639831543,-1.0846562385559082],[2.5396838188171387,-0.31746014952659607]],[[2.4390242099761963,0.8130078315734863],[-1.3821134567260742,-2.6829261779785156]],[[2.793834924697876,-2.6011569499969482],[0.19267812371253967,2.1194608211517334]],[[0.8003767132759094,1.553672432899475],[0.5649716258049011,-2.8248589038848877]]]
        # )


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
        # for i in range(3):
        #     F += torch.matmul(x[:, i].unsqueeze(-1), self.grad_undeformed_sample_weight[:, i].unsqueeze(1))
        # for i in range(3):


        # Calculate the Jacobian of the deformation
        # print(F.shape)
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

