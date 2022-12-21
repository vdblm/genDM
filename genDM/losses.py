from typing import Callable
import jax.numpy as jnp
import numpy as np

from dags import DAG


def output_loss(dag: DAG, target_var: int) -> Callable[[jnp.ndarray], jnp.ndarray]:
    start_idx, end_idx = np.sum(dag.var_dims[target_var]), np.sum(dag.var_dims[target_var + 1])

    def loss_fn(batch: jnp.ndarray) -> jnp.ndarray:
        return batch[:, start_idx:end_idx]

    return loss_fn
