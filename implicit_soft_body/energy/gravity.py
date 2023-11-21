import jax
import jax.numpy as jnp
import jax.nn as nn
from typing import Union
from .base import EnergyFunc

def gravity_energy(x:jax.Array, m:Union[float, jax.Array], g:Union[float,jax.Array]) -> jax.Array:
    """
    Gravity energy function.
    
    Args:
        x: (n, 3) array of positions or (n, 2). (x,y,z) or (x,y)
        m: (n,) array of masses or float if all masses are the same
        g: (1,) array of gravity acceleration
    Returns:
        Gravity energy.
    """
    # Compute energy.
    return jnp.sum(m * g * x[:,-1])

class GravityEnergy(EnergyFunc):
    def __init__(self, m:jax.Array, g:Union[float,jax.Array]=9.8, *args, **kwargs):
        """
        Args:
            m: (n,) array of masses
            g: (1,) array of gravity acceleration
        """
        self.__g = g
        self.__m = m
        gravity_energy_func = lambda x0, vx0,x, action: gravity_energy(x, self.__m, self.__g)
        super().__init__(gravity_energy_func, *args, **kwargs)

    def backward(self, *args, **kwargs):
        return None
    
    @property
    def g(self) -> Union[float,jax.Array]:
        return self.__g
    
    @property
    def m(self) -> jax.Array:
        return self.__m

if __name__=="__main__":
    sample_x = jnp.ones((10,2))
    sample_vx0 = jnp.ones((10,2))
    sample_x0 = jnp.ones((10,2))
    sample_m = jnp.ones((10,))
    sample_action = jnp.zeros((10,3))

    energy = gravity_energy(sample_x, sample_m, 9.8)
    energy_func = GravityEnergy(sample_m)
    energy_func.grad(sample_x0, sample_vx0, sample_x, sample_action)
    print("Everything passed. The output energy is", energy)
