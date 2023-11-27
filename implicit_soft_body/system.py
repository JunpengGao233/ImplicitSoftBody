import torch

# from .geometry.triangle import Triangle
from energy.neohookean import TriangleEnergy
from energy.friction import FrictionEnergy
from energy.collision import CollisionEnergy
from energy.spring import SpringEnergy
from energy.gravity import GravityEnergy
from energy.inertial import InertialEnergy


class MassSpringSystem:
    def __init__(
        self,
        vertices: torch.Tensor,
        springs: torch.Tensor,
        triangles: torch.Tensor,
        params: dict,
    ):
        self.springs = springs
        self.vertices = vertices
        self.triangles = triangles
        self.gravity_energy = GravityEnergy(params["mass"])
        self.spring_energy = SpringEnergy(params["k_spring"], params["l0"])
        self.neohookean_energy = TriangleEnergy(params["mu"], params["nu"])
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
        self.a = torch.ones(self.springs.shape[0])

    def add_spring(
        self, vertices: torch.Tensor, springs: torch.Tensor, triangles: torch.Tensor
    ):
        for triangle in triangles:
            self.triangles.append(triangle)
        for spring in springs:
            self.springs.append(spring)
        for vertex in vertices:
            self.vertices.append(vertex)

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            x = minimize(
                self.total_energy,
                self.x0,
                # method="Newton-CG",
                method="BFGS",
                # options={"disp": True},
            )
            return x
        self.x = f(x)

        return x

    def backward(self, x: torch.Tensor):
        f = jnp.negative(jax.grad(self.total_energy))
        dLdx = jax.grad(self.loss)(x)
        dfdx = jax.grad(f)(x)
        z = jax.np.linalg.solve(dfdx, dLdx)
        dfda = jax.grad(f)(self.a)
        dLda = -z.T @ dfda

        return dLda

    def loss(self, x: torch.Tensor):
        return -x[:, 0].mean()

    def total_energy(self, x: torch.Tensor):
        dt = self.dt

        springs_vertices = x[self.springs]
        potential_energy = 0
        potential_energy += self.gravity_energy.forward(x)
        potential_energy += self.spring_energy.forward(
            spring_vertices[:, 0], spring_vertices[:, 1], self.a
        )
        potential_energy += self.neohookean_energy.forward(
            self.x0[self.triangles], x[self.triangles]
        )

        external_energy = 0
        external_energy += self.collison_energy.forward(x)
        external_energy += self.friction_energy.forward(self.x0, x)

        inertial_energy = 0
        inertial_energy += self.inertial_energy.forward(x, self.x0, self.v0)

        return dt * dt * (potential_energy + external_energy) + 0.5 * inertial_energy


if __name__ == "__main__":
    x = torch.tensor([[0, 0], [2, 0], [1, 1]], dtype=torch.float32)
    triangles = torch.tensor([[0, 1, 2]])
    springs = torch.tensor([[0, 2], [1, 2]])
    params = {
        "mass": 1,
        "k_spring": 1,
        "l0": torch.sqrt(2),
        "mu": 1,
        "nu": 0.3,
        "k_collision": 1,
        "k_friction": 1,
        "epsilon": 0.01,
        "dt": 0.01,
    }
    system = MassSpringSystem(x, springs, triangles, params)
    system.forward(x)
