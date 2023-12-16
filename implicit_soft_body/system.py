import torch

# from .geometry.triangle import Triangle
from energy.neohookean import TriangleEnergy
from energy.friction import FrictionEnergy
from energy.collision import CollisionEnergy
from energy.spring import SpringEnergy
from energy.gravity import GravityEnergy
from energy.inertial import InertialEnergy

from network import MLP
import Sim


class MassSpringSystem:
    def __init__(
        self,
        vertices: torch.Tensor,
        springs: torch.Tensor,
        triangles: torch.Tensor,
        params: dict,
        a: torch.nn.Module = None,
    ):
        self.springs = springs
        self.vertices = vertices
        self.triangles = triangles
        self.l0 = params["l0"]
        triangles_vertices = vertices[triangles]
        self.gravity_energy = GravityEnergy(params["mass"])
        self.spring_energy = SpringEnergy(self.l0, params["k_spring"])
        self.neohookean_energy = TriangleEnergy(params["mu"], params["nu"], triangles_vertices)
        self.collison_energy = CollisionEnergy(params["k_collision"])
        self.friction_energy = FrictionEnergy(
            params["k_friction"], params["epsilon"], params["dt"]
        )
        self.inertial_energy = InertialEnergy(params["mass"], params["dt"])
        self.x = self.vertices
        self.x0 = self.vertices
        self.v = torch.zeros_like(self.x)
        self.v0 = torch.zeros_like(self.x)
        self.dt = params["dt"]
        # self.a = (0.5 * torch.ones_like(self.l0)).requires_grad_() if a is None else a
        self.max_iter = params["max_iter"]

    def add_spring(
        self, vertices: torch.Tensor, springs: torch.Tensor, triangles: torch.Tensor
    ):
        for triangle in triangles:
            self.triangles.append(triangle)
        for spring in springs:
            self.springs.append(spring)
        for vertex in vertices:
            self.vertices.append(vertex)

    def forward(self, x0: torch.Tensor, v0: torch.Tensor, a: torch.Tensor):
        x, v = Sim.DiffSim.apply(x0, v0, a, self.dt, self.max_iter, self)
        return x, v

    def loss(self, x: torch.Tensor):
        return -x[:, 0].mean()

    def total_energy(self, x0:torch.Tensor, x: torch.Tensor, v0: torch.Tensor, a: torch.Tensor):
        dt = self.dt

        spring_vertices = x[self.springs]
        potential_energy = 0
        potential_energy += self.gravity_energy.forward(x)
        potential_energy += self.spring_energy.forward(
            spring_vertices[..., 0], spring_vertices[..., 1], a
        )
        potential_energy += self.neohookean_energy.forward(
            x0[self.triangles], x[self.triangles]
        )

        external_energy = 0
        external_energy += self.collison_energy.forward(x)
        external_energy += self.friction_energy.forward(x0, x)

        inertial_energy = 0
        inertial_energy += self.inertial_energy.forward(x, x0, v0)
        return dt * dt * (potential_energy + external_energy) + 0.5 * inertial_energy


if __name__ == "__main__":
    x = torch.tensor([[0, 0], [2, 0], [1, 1]], dtype=torch.float32)
    triangles = torch.tensor([[0, 1, 2]])
    springs = torch.tensor([[0, 2], [1, 2]])
    params = {
        "mass": 1,
        "k_spring": 1,
        "l0": torch.sqrt(torch.tensor(2)),
        "mu": 1,
        "nu": 0.3,
        "k_collision": 1,
        "k_friction": 1,
        "epsilon": 0.01,
        "dt": 0.01,
        "max_iter": 100,
    }
    system = MassSpringSystem(x, springs, triangles, params)
    system.forward(x)
    system.backward(x)
