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
        if (self.config.optimizer == 'adam' or self.config.optimizer == 'adam_weight') and \
                self.config.momentums is not None:
            args['b1'] = self.config.momentums[0]
            args['b2'] = self.config.momentums[1]
        self.optimizer = OPTIMIZERS[self.config.optimizer](**args)
        if self.config.grad_clip is not None:
            self.optimizer = optax.chain(optax.clip(self.config.grad_clip), self.optimizer)