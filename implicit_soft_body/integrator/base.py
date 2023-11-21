from abc import abstractmethod, ABC
import jax
import jax.numpy as jnp
from typing import Callable, Union, Tuple, Optional


class Integrator(ABC):
    def __init__(
        self,
        mass: jax.Array,
        spring: Tuple[jax.Array, jax.Array],
        element: Tuple[jax.Array, jax.Array, jax.Array],
        fix_nodes: Optional[jax.Array] = None,
    ) -> None:
        self.__mass: jax.Array = mass
        self.__spring: Tuple[jax.Array, jax.Array] = spring
        assert self.__spring[0].shape == self.__spring[1].shape
        self.__element: Tuple[jax.Array, jax.Array, jax.Array] = element
        assert self.__element[0].shape == self.__element[1].shape
        assert self.__element[0].shape == self.__element[2].shape
        if fix_nodes is None:
            self.__fix_nodes = jnp.zeros(self.__mass.shape[0], dtype=jnp.bool_)
        else:
            self.__fix_nodes: Optional[jax.Array] = fix_nodes

    @abstractmethod
    def step(self,x:jax.Array, vx:jax.Array, action: jax.Array, dt: float) -> None:
        pass

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
    
    @property
    def fix_nodes(self) -> jax.Array:
        return self.__fix_nodes
    

