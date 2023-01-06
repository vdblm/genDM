import jax.random
import jax.numpy as jnp

from util.commons import Variables, Params
from typing import List

from util.optimizer import Optimizer
from flax.training import train_state


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


class TrainState(train_state.TrainState):
    stats: Variables
    init_stats: Variables
    init_params: Params

    def init_opt_state(self):  # Initializes the optimizer state. TODO make sure it's correct
        new_opt_state = self.tx.init(self.params)
        return self.replace(opt_state=new_opt_state)

    def init_param_state(self):
        return self.replace(stats=self.init_stats, params=self.init_params)


def create_state(rng, model_cls, model_config, optimizer_config, input_shapes: List):
    model = model_cls(model_config)
    optimizer = Optimizer(optimizer_config)
    variables = model.init(rng, *[jnp.ones(i) for i in input_shapes])
    stats, params = variables.pop("params")
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        stats=stats,
        tx=optimizer.optimizer,
        init_stats=stats,
        init_params=params
    )
    return state
