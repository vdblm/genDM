import optax

OPTIMIZERS = {
    'adam': optax.adam,
    'sgd': optax.sgd,
    'adagrad': optax.adagrad,
    'adam_weight': optax.adamw
}


class Optimizer:
    """Wrapped optimizer for optax."""

    def __init__(self, config):
        self.config = config  # Has keys: optimizer, learning_rate
        self.learning_rate = self.config.learning_rate
        args = {'learning_rate': self.learning_rate}
        if self.config.optimizer == 'adam_weight':
            args['weight_decay'] = self.config.weight_decay
        self.optimizer = OPTIMIZERS[self.config.optimizer](**args)
