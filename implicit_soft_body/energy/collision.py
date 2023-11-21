import jax 
import jax.numpy as jnp
import jax.nn as nn
from .base import EnergyFunc
from typing import Union


# Collision energy here refers to the collision energy between the ground and the soft body
#  (the coordinates of the ground is z=pos_wall in 3D and y=pos_wall in 2D)

def collision_energy(x:jax.Array, k:Union[jax.Array, float], pos_wall:Union[jax.Array, float] = 0.0) -> jax.Array:
    """
    Collision energy function.
    
    Args:
        x: (n, 3) array of positions or (n, 2). (x,y,z) or (x,y)
        k: Collision energy coefficient
        pos_wall: Position of the wall
    Returns:
        Collision energy.
    """
    # Compute energy.
    return 0.5 * k * jnp.sum(nn.relu(-x[:,-1] + pos_wall)**2)

class CollisionEnergy(EnergyFunc):
    def __init__(self, k:Union[jax.Array, float], pos_wall:Union[jax.Array, float] = 0.0, *args, **kwargs):
        """
        Args:
            k: Collision energy coefficient
        """
        self.__k:Union[jax.Array, float] = k
        self.__pos_wall:Union[jax.Array, float] = pos_wall
        collision_energy_func = lambda x0, vx0, x, action: collision_energy(x, k, pos_wall)
        super().__init__(collision_energy_func, *args, **kwargs)

    def backward(self, *args, **kwargs):
        return None
    
    @property
    def k(self) -> Union[jax.Array, float]:
        return self.__k
    
    @property
    def pos_wall(self) -> Union[jax.Array, float]:
        return self.__pos_wall 
    

if __name__=="__main__":
    sample_x0 = -jnp.ones((10,3))
    sample_vx0 = jnp.zeros((10,3))
    sample_x = jnp.zeros((10,3))
    sample_action = jnp.zeros((10))
    
    energy = collision_energy(sample_x, 20000, 0.0)
    energy_func = CollisionEnergy(20000)
    energy_func.grad(sample_x0, sample_vx0, sample_x, sample_action)
    print("Everything passed", energy)