from collections import OrderedDict
from typing import Callable, Sequence, Tuple, Dict

import flax.linen as nn

from util.dags import DAG

import jax.numpy as jnp

from typing import Any
import optax

PRNGKey = Any
Params = Any
Variables = Any
OptState = optax.OptState

Shape = Tuple[int, ...]
Dtype = Any
Array = Any


class MLP(nn.Module):
    features: Sequence[int]
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1)
    activation: str = 'relu'

    @nn.compact
    def __call__(self, x):
        if self.activation == 'relu':
            activation_fn = nn.relu
        elif self.activation == 'gelu':
            activation_fn = nn.gelu
        elif self.activation == 'tanh':
            activation_fn = nn.tanh
        else:
            raise ValueError(f"Expected a valid activation, got {self.activation}")
        for i, feat in enumerate(self.features):
            x = nn.Dense(features=feat, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
            if i != len(self.features) - 1:
                x = activation_fn(x)
        return x


class CausalGenerator(nn.Module):
    """A generative model that follows a causal graph.
    For the observational data, we append the latent variables and parents. For the interventional data, we use the
    interventional model to generate.
    """
    causal_graph: DAG
    obs_models: dict

    @nn.compact
    def __call__(self, z, interventions: Dict[int, nn.Module]):
        """Generates a sample from the causal generator."""
        latent_dims = self.causal_graph.latent_dims
        assert z.shape[1] == sum(latent_dims), "z must have the correct latent dimension"
        assert len(z.shape) == 2, "z must be a batch of latent vectors"
        outputs = OrderedDict()
        for var, pars in self.causal_graph.graph.items():
            latents = [i for i in self.causal_graph.graph[var] if i < 0]
            observed = [i for i in self.causal_graph.graph[var] if i >= 0]
            model_inputs = (
                    [z[:, sum(latent_dims[abs(i) - 1]):sum(latent_dims[abs(i)])] for i in latents] +
                    [outputs[i] for i in observed]
            )
            model_inputs = jnp.concatenate(model_inputs, axis=1)
            model = self.obs_models[var] if var not in interventions else interventions[var]
            outputs[var] = model(model_inputs)

        outputs = [outputs[i] for i in outputs]
        return jnp.concatenate(outputs, axis=1)
