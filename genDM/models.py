# Copyright (c) 2022, Vahid Balazadeh Meresht
# MIT License

import flax
import flax.linen as nn
import jax.lax
import jax.numpy as jnp

from util.utils import ConfigDict

from util.commons import MLP


class RatioModel(nn.Module):
    """A model for the ratio r_{phi}(v) = q_{phi}(v) / p_{true}(v). It implements the
    constraint E_{v ~ p_{true}}[r_{phi}(v)] = 1 using Batch level normalization, i.e.,
    r_{phi}(v) = exp(f_{phi}(v)) / (1/m sum_{i=1}^m exp(f_{phi}(v_i))).
    """

    # model configuration. Has the following keys: 'features', 'activation', 'momentum', 'stable_eps'
    config: ConfigDict

    @nn.compact
    def __call__(self, x, train: bool = False, update_stats: bool = True):
        is_initialized = self.has_variable("batch_stats", "mean")

        # running average mean
        ra_mean = self.variable("batch_stats", "mean",
                                lambda s: jnp.ones(s),
                                self.config['features'][-1])  # type: flax.core.scope.Variable

        # create the model using config
        f = MLP(features=self.config['features'], activation=self.config['activation'])(x)

        exp_f = nn.sigmoid(f)  # TODO we may use exp(f) instead of sigmoid(f) to avoid zero grads

        mean = ra_mean.value

        if train:
            mean = jnp.mean(exp_f, axis=0)  # TODO use pmean with `batch` axis_name if we use vmap
            if is_initialized and update_stats:
                ra_mean.value = self.config['momentum'] * ra_mean.value + (1.0 - self.config['momentum']) * mean

        return exp_f / (mean + self.config['stable_eps'])  # normalize the model


class CPM(nn.Module):
    """Conditional Probability Model. It returns log(p_{theta}(t|x)).
    For now, we consider a simple Gaussian model with fixed variance.
    """

    # model configuration. Has the following keys: 'condition_dim', 'decision_dim', 'features', 'activation', 'variance'
    config: ConfigDict

    @nn.compact
    def __call__(self, conditions, decision):
        if self.config['variance'] is None:
            log_sigma = self.param('log_sigma', nn.initializers.zeros, (1, self.config['decision_dim']))
        else:
            log_sigma = jnp.log(self.config['variance'] * jnp.ones((1, self.config['decision_dim'])))
        assert self.config['features'][0] == self.config['condition_dim']
        assert self.config['features'][-1] == self.config['decision_dim']
        mu_x = MLP(features=self.config['features'], activation=self.config['activation'])(conditions)
        return (
                - 0.5 * jnp.sum((decision - mu_x) ** 2 / jnp.exp(log_sigma), axis=1, keepdims=True)
                - 0.5 * jnp.sum(log_sigma, axis=1, keepdims=True)
                - 0.5 * self.config['decision_dim'] * jnp.log(2 * jnp.pi)
        )


def test_ratio_model():
    from util.utils import create_state, TrainState, PRNGSequence
    import tqdm
    model_conf = ConfigDict(
        {
            'features': [1, 32, 32, 1],
            'activation': 'tanh',
            'momentum': 0.9,
            'stable_eps': 0.
        }
    )

    opt_conf = ConfigDict(
        {
            'learning_rate': 1e-2,
            'optimizer': 'adam',
            'epochs': 2000
        }
    )
    rng = PRNGSequence(2539)
    data = jax.random.normal(next(rng), (100, 1)) * 2 + 2

    model_state = create_state(next(rng), RatioModel, model_conf, opt_conf, [(1, 1)])

    @jax.jit
    def _update_step(state: TrainState, batch):
        def loss_fn(params):
            r, avg = state.apply_fn(
                {'params': params, **state.state},
                batch,
                mutable=list(state.state.keys()),
                train=True)
            return - jnp.mean(r * batch), (avg, {'expected_r': jnp.mean(r)})

        (loss, (stats, out)), grad = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grad, state=stats)
        out['loss'] = loss
        out['mean_raw'] = new_state.state['batch_stats']['mean']
        return new_state, out

    with tqdm.tqdm(total=opt_conf.epochs) as pbar:
        for epoch in range(opt_conf.epochs):
            model_state, output = _update_step(model_state, data)
            logging_str = ' '.join(['{}: {:.4f}'.format(k, v.item()) for (k, v) in output.items()])
            pbar.set_postfix_str(logging_str)
            pbar.update(1)
    assert jnp.isclose(-output['loss'], jnp.max(data),
                       atol=1e-2), "The model should be able to learn the max of the data"


def test_ratio_model_constraint():
    # TODO for the constrained case, it is highly sensitive to the lagrange learning rate and number of epochs
    # TODO The hyper-parameters are also sensitive to the scale of loss
    # TODO Also, it seems stable_eps should be set to 0.0
    from util.utils import create_state, TrainState, PRNGSequence
    import tqdm
    model_conf = ConfigDict(
        {
            'features': [1, 32, 32, 1],
            'activation': 'tanh',
            'momentum': 0.9,
            'stable_eps': 0.
        }
    )

    opt_conf = ConfigDict(
        {
            'learning_rate': 1e-2,
            'optimizer': 'adam',
            'epochs': 1000,
            'lagrange_lr': 0.5,
            'max_lambda': 100.,
            'alpha': 0.1
        }
    )

    rng = PRNGSequence(2539)
    data = jax.random.normal(next(rng), (100, 1)) * 2 + 2

    model_state = create_state(next(rng), RatioModel, model_conf, opt_conf, [(1, 1)])
    lambda_param = 0.

    @jax.jit
    def _update_step(state: TrainState, lagrange_multiplier, batch):
        def loss_fn(params, lagrange):
            r, avg = state.apply_fn(
                {'params': params, **state.state},
                batch,
                mutable=list(state.state.keys()),
                train=True)
            target_loss = - jnp.mean(r * (batch ** 2))
            kl = jnp.mean(r * jnp.log(r))
            return target_loss + lagrange * (kl - opt_conf.alpha), (avg, {'expected_r': jnp.mean(r),
                                                                          'kl': kl})

        (loss, (stats, out)), (param_grad, lagrange_grad) = (
            jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1))(state.params,
                                                                      lagrange_multiplier)
        )
        new_state = state.apply_gradients(grads=param_grad, state=stats)
        new_lagrange_params = lagrange_multiplier + opt_conf.lagrange_lr * lagrange_grad
        # Ensure non-negative lambda and lambda <= max_lambda
        new_lagrange_params = jnp.maximum(new_lagrange_params, 0)
        new_lagrange_params = jnp.minimum(new_lagrange_params, opt_conf.max_lambda)
        out['loss'] = loss
        out['mean_raw'] = new_state.state['batch_stats']['mean']
        out['lagrange'] = new_lagrange_params
        return new_state, new_lagrange_params, out

    with tqdm.tqdm(total=opt_conf.epochs) as pbar:
        for epoch in range(opt_conf.epochs):
            model_state, lambda_param, output = _update_step(model_state, lambda_param, data)
            logging_str = ' '.join(['{}: {:.9f}'.format(k, v.item()) for (k, v) in output.items()])
            pbar.set_postfix_str(logging_str)
            pbar.update(1)

    assert jnp.max(data ** 2) > -output['loss'] > jnp.mean(data ** 2), "The result should be between average and max"
    assert jnp.isclose(output['kl'], opt_conf.alpha, atol=1e-2), "The final KL should be close to alpha"
