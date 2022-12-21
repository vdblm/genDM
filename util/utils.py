from abc import ABC

import jax.random
import jax.numpy as jnp

from types import Params, Variables, OptState
from typing import Callable

from optimizer import Optimizer
from flax import struct


class ConfigDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# adapted from https://github.com/unstable-zeros/tasil
class PRNGSequence:
    def __init__(self, key_or_seed):
        if isinstance(key_or_seed, int):
            key_or_seed = jax.random.PRNGKey(key_or_seed)
        elif (hasattr(key_or_seed, "shape") and (not key_or_seed.shape) and
              hasattr(key_or_seed, "dtype") and key_or_seed.dtype == jnp.int32):
            key_or_seed = jax.random.PRNGKey(key_or_seed)
        self._key = key_or_seed

    def __next__(self):
        k, n = jax.random.split(self._key)
        self._key = k
        return n


class TrainState(struct.PyTreeNode, ABC):
    step: int
    apply_fn: Callable
    params: Params
    state: Variables
    opt: Optimizer
    opt_state: OptState

    def apply_gradients(self, *, grads, **kwargs):
        new_params, new_opt_state = self.opt.opt_update(grads, self.opt_state, self.params)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, opt, **kwargs):
        opt_state = opt.opt_init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            opt=opt,
            opt_state=opt_state,
            **kwargs,
        )


def create_state(rng, model_cls, model_config, optimizer_config, input_shape):
    model = model_cls(model_config)
    optimizer = Optimizer(optimizer_config)
    variables = model.init(rng, jnp.ones(input_shape))
    state, params = variables.pop("params")
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        state=state,
        opt=optimizer,
    )
    return state

    # def _get_treatment_parents(self, batch: jnp.ndarray) -> jnp.ndarray:
    #     variables = self.dag.get_obs_parents(self.gc.treatment) + [self.gc.treatment]
    #     idx = np.array(
    #         [np.arange(np.sum(self.dag.var_dims[:i]), np.sum(self.dag.var_dims[:(i + 1)])) for i in variables]
    #     ).flatten()
    #     return batch[:, idx]
