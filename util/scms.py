# Copyright (c) 2022, Vahid Balazadeh Meresht
# MIT License

from typing import Callable
from util.commons import PRNGKey
import jax.numpy as jnp
import numpy as np
import jax

from util.dags import DAG, gen_dags


class SCM:
    """Describes the structural causal model that generates the data.
    Args:
        name (str): Name of the data generating process.
        dag (DAG): DAG structure of the SCM.
        generate_func (Callable[[PRNGKey, int], np.ndarray]): Function that generates the data.

    Attributes:
        data (np.ndarray): Data generated by the SCM.
    """

    def __init__(self, name: str, dag: DAG, generate_func: Callable[[PRNGKey, int], np.ndarray],
                 treatment_var: int, target_var: int):
        self.name = name
        self.generate_func = generate_func
        self.dag = dag

        self.data = None
        self.treatment, self.target = treatment_var, target_var

    def generate(self, key: PRNGKey, n_samples: int) -> np.ndarray:
        data = self.generate_func(key, n_samples)
        self.data = data
        return data


def linear_backdoor():
    def generate(key: PRNGKey, n_samples: int) -> np.ndarray:
        keys = jax.random.split(key, 3)
        x = 1 + jax.random.normal(key=keys[0], shape=(n_samples, 1))
        t = x + 2 + jax.random.normal(keys[1], shape=(n_samples, 1))
        y = (- x + 3 * t + jax.random.normal(keys[2], shape=(n_samples, 1))) / 10
        data = np.array(jnp.concatenate([x, t, y], axis=-1), dtype=jnp.float32)
        return data

    dag = gen_dags('backdoor', var_dims=[1, 1, 1])
    scm = SCM('linear_backdoor', dag, generate, treatment_var=1, target_var=2)
    return scm


def gen_scm(key: str) -> SCM:
    return {
        'linear_backdoor': linear_backdoor(),
    }[key]

