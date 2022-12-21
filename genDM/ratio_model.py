# Copyright (c) 2022, Vahid Balazadeh Meresht
# MIT License

import flax
import flax.linen as nn
import jax.numpy as jnp

from utils import ConfigDict

from models import MLP


class RatioModel(nn.Module):
    """A model for the ratio r_{phi}(v) = q_{phi}(v) / p_{true}(v). It implements the
    constraint E_{v ~ p_{true}}[r_{phi}(v)] = 1 using Batch level normalization, i.e.,
    r_{phi}(v) = exp(f_{phi}(v)) / (1/m sum_{i=1}^m exp(f_{phi}(v_i))).
    """

    # model configuration. Has the following keys: 'features', 'activation', 'momentum', 'stable_eps'
    config: ConfigDict

    @nn.compact
    def __call__(self, x, train: bool = False):
        is_initialized = self.has_variable("batch_stats", "mean")

        # running average mean
        ra_mean = self.variable("batch_stats", "mean", lambda: jnp.ones, x.shape[1:])  # type: flax.core.scope.Variable

        exp_f = jnp.exp(
            MLP(features=self.config.features, activiation=self.config.activation)(x))  # create the model using config

        if not train:
            mean = ra_mean.value
        else:
            mean = jnp.mean(exp_f, axis=0)  # TODO use pmean with `batch` axis_name if we use vmap
            if is_initialized:
                ra_mean.value = self.config.momentum * ra_mean.value + (1.0 - self.config.momentum) * mean

        # normalize the model
        return exp_f / (mean + self.config.stable_eps)
