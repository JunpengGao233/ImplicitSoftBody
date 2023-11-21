import jax
import jax.numpy as jnp
from typing import Tuple, Optional, List, Union
from .base import Integrator
from ..energy.base import EnergyFunc



def __implicit_euler_energy(m:jax.Array, x0: jax.Array, vx0: jax.Array, x:jax.Array, dt:Union[jax.Array,float]) -> jax.Array:

    return m * ((x0 + dt * vx0 - x)) ** 2 

class BackwardEuler(Integrator):
    def __init__(
        self,
        mass: jax.Array,
        spring: Tuple[jax.Array, jax.Array],
        element: Tuple[jax.Array, jax.Array, jax.Array],
        fix_nodes: Optional[jax.Array] = None,
        energy_funcs: List[EnergyFunc] = [],
        max_iter: int = 100,
    ):
        self.__energy_funcs = energy_funcs
        self.__max_iter = max_iter
        super().__init__(mass=mass, spring=spring, element=element, fix_nodes=fix_nodes)


    def forward(self, x0: jax.Array, vx0: jax.Array, x:jax.Array, action: jax.Array, dt:Union[jax.Array, float]) -> jax.Array:
        energy = __implicit_euler_energy(self.mass, x0, vx0, x, dt)
        for energy_func in self.__energy_funcs:
            energy += energy_func(x0, vx0, x, action)

        return energy

    
    def step(self, x0:jax.Array, vx0:jax.Array, action: jax.Array, dt: float ) -> None:
        pass

    def add_energy_func(self, energy_func: EnergyFunc) -> None:
        self.__energy_funcs.append(energy_func)
    
    def remove_energy_func(self, energy_func: EnergyFunc) -> None:
        self.__energy_funcs.remove(energy_func) # TODO: not sure whether remove operation works or not

    @property
    def energy_funcs(self) -> List[EnergyFunc]:
        return self.__energy_funcs
    
    