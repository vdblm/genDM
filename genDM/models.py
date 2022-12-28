# Copyright (c) 2022, Vahid Balazadeh Meresht
# MIT License

import flax
import flax.linen as nn
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
        ra_mean = self.variable("batch_stats", "mean", lambda s: jnp.ones(s),
                                x.shape[1:])  # type: flax.core.scope.Variable

        exp_f = jnp.exp(
            MLP(features=self.config['features'], activation=self.config['activation'])(
                x))  # create the model using config

        if not train:
            mean = ra_mean.value
        else:
            mean = jnp.mean(exp_f, axis=0)  # TODO use pmean with `batch` axis_name if we use vmap
            if is_initialized and update_stats:
                ra_mean.value = self.config['momentum'] * ra_mean.value + (1.0 - self.config['momentum']) * mean

        # normalize the model
        return exp_f / (mean + self.config['stable_eps'])


class CPM(nn.Module):
    """Conditional Probability Model. It returns log(p_{theta}(t|x)).
    For now, we consider a simple Gaussian model with fixed variance.
    """

    # model configuration. Has the following keys: 'condition_dim', 'decision_dim', 'features', 'activation'
    config: ConfigDict

    @nn.compact
    def __call__(self, conditions, decision):  # TODO double check
        log_sigma = self.param('log_sigma', nn.initializers.zeros, (1, self.config['decision_dim']))
        assert self.config['features'][0] == self.config['condition_dim']
        assert self.config['features'][-1] == self.config['decision_dim']
        mu_x = MLP(features=self.config['features'], activation=self.config['activation'])(conditions)
        return (
                - 0.5 * jnp.sum((decision - mu_x) ** 2 / jnp.exp(log_sigma), axis=1, keepdims=True)
                - 0.5 * jnp.sum(log_sigma, axis=1, keepdims=True)
                - 0.5 * self.config['decision_dim'] * jnp.log(2 * jnp.pi)
        )
