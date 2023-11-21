from abc import abstractmethod, ABC
import jax
from jax import grad
from typing import Any, Callable


class EnergyFunc(ABC):
    def __init__(
        self,
        func: Callable[
            [jax.Array, jax.Array, jax.Array, jax.Array], jax.Array
        ],
        *args,
        **kwargs,
    ):
        self.__func = func
        self.__grad = grad(self.__func, *args, **kwargs)

    def forward(
        self,
        x0: jax.Array,
        vx0: jax.Array,
        x: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        """
        Compute energy with given position, velocity and action.
        """
        return self.__func(x0, vx0, x, action)

    @abstractmethod
    def backward(self, *args, **kwargs):

        pass

    def __call__(self, x0: jax.Array, vx0: jax.Array, x: jax.Array, action: jax.Array) -> jax.Array:
        return self.forward(x0, vx0, x, action)
    
    @property
    def func(self):
        return self.__func

    @property
    def grad(self):
        return self.__grad
