import optax

OPTIMIZERS = {
    'adam': optax.adam,
    'sgd': optax.sgd,
    'adagrad': optax.adagrad,
}


class Optimizer:
    """Wrapped optimizer for optax."""

    def __init__(self, config):
        self.config = config  # Has keys: optimizer, learning_rate
        self.learning_rate = self.config.learning_rate
        self.optimizer = OPTIMIZERS[self.config.optimizer](learning_rate=self.learning_rate)
