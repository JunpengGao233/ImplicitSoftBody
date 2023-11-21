import jax
import jax.numpy as jnp

from typing import Union, Optional, Tuple


class MassSpringSystem:
    def __init__(
        self,
        mass: jax.Array,
        spring: Tuple[jax.Array, jax.Array],
        element: Tuple[jax.Array, jax.Array, jax.Array],
        x0: Optional[jax.Array] = None,
        v0: Optional[jax.Array] = None,  # dt might be changable here
        fix_nodes:Optional[jax.Array] = None,
        rsi: Optional[jax.Array] = None,  # not implemented yet
    ) -> None:
        
        self.__mass = mass
        self.__spring = spring
        assert self.__spring[0].shape == self.__spring[1].shape
        self.__element = element
        assert self.__element[0].shape == self.__element[1].shape
        assert self.__element[0].shape == self.__element[2].shape
        self.__x0 = x0
        self.__v0 = v0
        self.__fix_nodes = fix_nodes

    @property
    def mass(self) -> jax.Array:
        return self.__mass
    
    @property
    def num_nodes(self) -> int:
        return self.__mass.shape[0]
    
    @property
    def spring(self) -> Optional[jax.Array]:
        return self.__spring
    
    @property
    def num_springs(self) -> Optional[int]:
        return self.__spring[0].shape[0]
    
    @property
    def element(self) -> Tuple[jax.Array, jax.Array, jax.Array]:
        return self.__element
    
    @property
    def num_elements(self) -> int:
        return self.__element[0].shape[0]
    


