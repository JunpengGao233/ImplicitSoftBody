import jax
import jax.numpy as jnp

from .geometry.triangle import Triangle
from .energy.neohookean import TriangleEnergy
from .energy.friction import FrictionEnergy
from .energy.collision import CollisionEnergy
from .energy.spring import SpringEnergy
from .energy.gravity import GravityEnergy
from .energy.inertial import InertialEnergy


class MassSpringSystem:
    def __init__(
        self,
        vertices: jax.Array,
        springs: jax.Array,
        triangles: jax.Array,
        params: dict,
    ):
        self.springs = springs
        self.vertices = vertices
        self.triangles = triangles
        self.gravity_energy = GravityEnergy(params["mass"])
        self.spring_energy = SpringEnergy(params["k"], params["l0"])
        self.neohookean_energy = TriangleEnergy(params["mu"], params["nu"])
        self.collison_energy = CollisionEnergy(params["k_collision"])
        self.friction_energy = FrictionEnergy(
            params["k_friction"], params["epsilon"], params["dt"]
        )
        self.inertial_energy = InertialEnergy(params["mass"], params["dt"])
        self.x = self.vertices
        self.x0 = self.vertices
        self.v = jnp.zeros_like(self.x)
        self.v0 = jnp.zeros_like(self.x)
        self.dt = params["dt"]
        self.a = jnp.ones(self.springs.shape[0])

    def add_spring(self, vertices: jax.Array, springs: jax.Array, triangles: jax.Array):
        for triangle in triangles:
            self.triangles.append(triangle)
        for spring in springs:
            self.springs.append(spring)
        for vertex in vertices:
            self.vertices.append(vertex)

    def forward(self, x: jax.Array):
        def f(x):
            x = jax.lax.stop_gradient(x)
            x = jax.scipy.optimize.minimize(
                self.total_energy,
                self.x0,
                method="newton-cg",
                options={"disp": True},
            )
            return x
        self.x = f(x)

        return x
    
    def backward(self, x: jax.Array):
        f =  jnp.negative(jax.grad(self.total_energy))
        dLdx = jax.grad(self.loss)(x)
        dfdx = jax.grad(f)(x)
        z = jax.np.linalg.solve(dfdx, dLdx)
        dfda = jax.grad(f)(self.a)
        dLda = - z.T @ dfda

        return dLda

    def loss (self, x: jax.Array):
        return - x[:, 0].mean()

    def total_energy(self, x: jax.Array):
        dt = self.dt

        spring_vertices = jnp.take(x, self.springs, axis=0)  # (n, 2, 3)
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

        return dt* dt * (potential_energy + external_energy) + 0.5 * inertial_energy
