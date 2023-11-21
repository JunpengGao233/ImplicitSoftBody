import jax
import jax.numpy as jnp
from typing import Union
from .base import EnergyFunc


def spring_energy(x:jax.Array,y:jax.Array,l0:jax.Array, k:float) -> jax.Array:
    """
    Spring energy function.
    
    Args:
        x: (n, 3) array of positions or (n, 2)
        y: (n, 3) array of positions or (n, 2)
        a: (n) Stretch of Spring
        l0: Spring rest length.
        k: Spring stiffness coefficient
    Returns:
        Spring energy.
    """
    # Compute spring length.
    l = jnp.linalg.norm(x - y, axis=-1)
    # Compute energy.
    return 0.5 * k * jnp.sum((l - l0)**2)

class SpringEnergy(EnergyFunc):
 
    def __init__(self, active_idx:jax.Array, l0:jax.Array, idx_0:jax.Array, idx_1:jax.Array, k:Union[jax.Array,float], *args, **kwargs):
        self.__active_idx:jax.Array = active_idx
        self.__l0:jax.Array = l0
        self.__idx_0:jax.Array = idx_0
        self.__idx_1:jax.Array = idx_1
        self.__k:Union[jax.Array,float] = k
        spring_energy_func = lambda x0, vx0, x, action: spring_energy(x[idx_0,:], x[idx_1,:], l0[active_idx] * action, k)
        super().__init__(spring_energy_func, *args, **kwargs)

    def backward(self, *args, **kwargs):
        return None
    
    @property
    def active_idx(self) -> jax.Array:
        return self.__active_idx

    @property
    def l0(self) -> jax.Array:
        return self.__l0
    
    @property
    def idx_0(self) -> jax.Array:
        return self.__idx_0
    
    @property
    def idx_1(self) -> jax.Array:
        return self.__idx_1
    
    @property
    def k(self) -> Union[jax.Array,float]:
        return self.__k
    

if __name__=="__main__":
    sample_x0 = jnp.zeros((10,3))
    sample_vx0 = jnp.zeros((10,3))
    sample_x = jnp.zeros((10,3))
    sample_action = jnp.zeros((9))
    
    energy = spring_energy(sample_x, sample_x0, 1.0, 20000)
    energy_func = SpringEnergy(jnp.arange(9).astype(jnp.int32), jnp.zeros(10,),jnp.ones(9,).astype(jnp.int32), jnp.arange(1,10).astype(jnp.int32), 20000)
    energy_func.grad(sample_x0, sample_vx0, sample_x, sample_action)
    print("Everything passed", energy)

