from typing import Union

import jax
import jax.numpy as jnp

from triangle import Triangle

class AssembledModel():
    def __init__(self, p0:jax.Array, v0:jax.Array, m:jax.Array, triangles, springs):       
        self.num_vertices = p0.shape[0]
        self.m = m
        self.v0 = v0
        self.springs = springs
        self.triangles = triangles