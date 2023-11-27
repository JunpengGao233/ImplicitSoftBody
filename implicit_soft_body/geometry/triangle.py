import jax
import jax.numpy as jnp


class Triangle:
    def __init__(self, node1:jax.Array, node2:jax.Array, node3:jax.Array):
        self.__node1 = node1
        self.__node2 = node2
        self.__node3 = node3
    
    @property
    def vertices(self):
        return jnp.stack([self.__node1, self.__node2, self.__node3], axis=0)