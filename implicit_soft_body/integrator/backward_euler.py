import jax
import jax.numpy as jnp
from typing import Tuple, Optional, List, Union
from .base import Integrator


class BackwardEuler(Integrator):
    def __init__(self, h):
        self.__h = h

    def forward(x0, v0):
        