# Copyright (c) 2022, Vahid Balazadeh Meresht
# MIT License

from typing import Callable
import jax.numpy as jnp
import numpy as np

from util.scms import SCM


def y2_loss(scm: SCM) -> Callable[[jnp.ndarray], jnp.ndarray]:
    start_idx, end_idx = np.sum(scm.dag.var_dims[:scm.target]), np.sum(scm.dag.var_dims[:(scm.target + 1)])

    def loss_fn(batch: jnp.ndarray) -> jnp.ndarray:
        raw_output = jnp.mean(batch[:, start_idx:end_idx] ** 2, axis=1, keepdims=True)
        return raw_output  # TODO maybe normalize it

    return loss_fn
