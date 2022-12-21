# Copyright (c) 2022, Vahid Balazadeh Meresht
# MIT License

from utils import ConfigDict
import optax

from types import Params, OptState
from typing import Tuple

OPTIMIZERS = {
    'adam': optax.adam,
    'sgd': optax.sgd,
    'adagrad': optax.adagrad,
}


class Optimizer:
    """Wrapped optimizer for optax."""

    def __init__(self, config: ConfigDict):
        self.config = config
        self.learning_rate = self.config.learning_rate
        self.optimizer = OPTIMIZERS[self.config.optimizer](learning_rate=self.learning_rate)

        self.opt_init = self.optimizer.init

    def opt_update(self, grads: Params, opt_state: OptState, params: Params) -> Tuple[Params, OptState]:
        state, update = self.optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, update), state
