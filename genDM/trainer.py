# Copyright (c) 2022, Vahid Balazadeh Meresht
# MIT License

from typing import Callable, Dict
from types import PRNGKey

import jax

import jax.numpy as jnp

from utils import ConfigDict, TrainState, create_state
from ratio_model import RatioModel
from dags import DAG, gen_dags
from losses import output_loss


class MinMaxTrainer:  # TODO Check for gpus
    """Training the generative Causal Decision-Making. It solves the following constrained optimization:
    min_{theta} max_{phi} E_{v ~ p(V)} [w_{theta}(t, x) · r_{phi}(v) · loss(v)]
    s.t. KL(q_{phi}(v) || p_{true}(v)) <= alpha <=> E_{v ~ p(V)}[log(r_{phi}(v)) · r_{phi}(v)] <= alpha
         E_{t, x ~ p(T, X)}[w_{theta}(t, x)] = 1
         E_{v ~ p(V)}[r_{phi}(v)] = 1

    We convert the above optimization problem to the following unconstrained optimization:
    min_{theta} max_{phi} min_{λ > 0}
    E_{v ~ p(V)} [w_{theta}(t, x) · r_{phi}(v) · loss(v)] - λ * (E_{v ~ p(V)}[log(r_{phi}(v)) · r_{phi}(v)] - alpha)
    """

    def __init__(self, global_config: ConfigDict, decision_model_config: ConfigDict, adversary_model_config: ConfigDict,
                 decision_optim_config: ConfigDict, adversary_optim_config: ConfigDict):
        self.gc = global_config  # Has keys: 'dag_name', 'var_dims', 'treatment', 'target_var', 'decision_dims',
        # 'lagrange_lr', 'alpha', 'max_lambda'
        self.dmc = decision_model_config
        self.amc = adversary_model_config
        self.doc = decision_optim_config
        self.aoc = adversary_optim_config

        self.dag = gen_dags(self.gc.dag_name, self.gc.var_dims)  # type: DAG
        self.loss = output_loss(self.dag, self.gc.target_var)  # type: Callable

        self.decision_state, self.adversary_state = None, None
        self._update_step = None
        self._eval_fn = None

        self.lagrange_multiplier = jnp.array([0.0])

    def initialize(self, rng: PRNGKey):
        dkey, akey = jax.random.split(rng)
        self.decision_state = create_state(dkey, RatioModel, self.dmc, self.doc,
                                           (1, self.dmc.features[0]))  # type: TrainState

        self.adversary_state = create_state(akey, RatioModel, self.amc, self.aoc,
                                            (1, self.amc.features[0]))  # type: TrainState

        @jax.jit
        def _update_step(key: PRNGKey, decision_state: TrainState, adversary_state: TrainState,
                         lagrange_multiplier: jnp.ndarray, batch: jnp.ndarray):
            def _loss_fn(decision_params, adversary_params, lagrange_param):
                w_theta, decision_mutable = decision_state.apply_fn({'params': decision_params, **decision_state.state},
                                                                    batch[:, self.gc.decision_dims],
                                                                    mutable=list(decision_state.state.keys()),
                                                                    train=True)
                r_phi, adversary_mutable = adversary_state.apply_fn(
                    {'params': adversary_params, **adversary_state.state},
                    batch,
                    mutable=list(adversary_state.state.keys()),
                    train=True)
                kl = jnp.mean(jnp.log(r_phi) * r_phi)
                target = jnp.mean(w_theta * r_phi * self.loss(batch))
                loss = target - lagrange_param * (kl - self.gc.alpha)
                return loss, target, kl, decision_mutable, adversary_mutable

            (
                (total_loss, avg_target, kl_constraint, decision_extra_states, adversary_extra_states),
                (decision_grads, adversary_grads, lagrange_grad)
            ) = jax.value_and_grad(_loss_fn, has_aux=True, argnums=(0, 1, 2))(decision_state.params,
                                                                              adversary_state.params,
                                                                              lagrange_multiplier)

            outputs = {'total_loss': total_loss, 'avg_target': avg_target, 'kl_constraint': kl_constraint}

            new_decision_state = decision_state.apply_gradients(grads=decision_grads, state=decision_extra_states)
            new_adversary_state = adversary_state.apply_gradients(grads=adversary_grads, state=adversary_extra_states)
            new_lagrange_multiplier = lagrange_multiplier - self.gc.lagrange_lr * lagrange_grad

            return new_decision_state, new_adversary_state, new_lagrange_multiplier, outputs

        @jax.jit  # TODO add the kl constraint to the eval function
        def _eval_fn(key: PRNGKey, decision_state: TrainState, adversary_state: TrainState, batch: jnp.ndarray):
            w_theta = decision_state.apply_fn({'params': decision_state.params, **decision_state.state},
                                              batch[:, self.gc.decision_dims],
                                              train=False)
            r_phi = adversary_state.apply_fn({'params': adversary_state.params, **adversary_state.state},
                                             batch,
                                             train=False)
            loss = jnp.mean(w_theta * r_phi * self.loss(batch))
            return loss

        self._update_step = _update_step
        self._eval_fn = _eval_fn

    def train_step(self, key: PRNGKey, batch: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        self.decision_state, self.adversary_state, new_lagrange_multiplier, outputs = self._update_step(
            key, self.decision_state, self.adversary_state, self.lagrange_multiplier, batch
        )

        # Ensure non-negative lambda and lambda <= max_lambda
        self.lagrange_multiplier = jnp.maximum(new_lagrange_multiplier, jnp.array([0.0]))
        self.lagrange_multiplier = jnp.minimum(self.lagrange_multiplier, jnp.array([self.gc.max_lambda]))
        return outputs

    def eval_step(self, key: PRNGKey, batch: jnp.ndarray):
        return self._eval_fn(key, self.decision_state, self.adversary_state, batch)
