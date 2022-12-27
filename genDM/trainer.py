# Copyright (c) 2022, Vahid Balazadeh Meresht
# MIT License

from typing import Callable, Dict, Tuple
from types import PRNGKey

import jax

import jax.numpy as jnp

from utils import ConfigDict, TrainState, create_state
from models import RatioModel, CPM


class MinMaxTrainer:  # TODO Check for gpus
    """Training the generative Causal Decision-Making. It solves the following constrained optimization:
    min_{theta} max_{phi} E_{v ~ p(V)} [r_{phi}(v) · loss(v)]
    s.t. KL(q_{phi}(v) || p_{true}(v|t) · pi_{theta}(t|x)) <= alpha
         E_{v ~ p(V)}[r_{phi}(v)] = 1
    """

    def __init__(self,
                 target_loss: Callable[[jnp.ndarray], jnp.ndarray],
                 log_p_true: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],  # args: conditions, decision
                 global_config: ConfigDict,
                 decision_model_config: ConfigDict, adversary_model_config: ConfigDict,
                 decision_optim_config: ConfigDict, adversary_optim_config: ConfigDict):
        # Has keys: 'batch_condition_dims', 'batch_decision_dims', 'lagrange_lr', 'alpha', 'max_lambda', 'inner_steps'
        self.gc = global_config
        self.dmc = decision_model_config
        self.amc = adversary_model_config
        self.doc = decision_optim_config
        self.aoc = adversary_optim_config

        self.log_p_true = log_p_true

        self.loss = target_loss

        self.decision_state, self.adversary_state = None, None
        self._update_step = None
        self._eval_fn = None

        self.lagrange_multiplier = jnp.array([0.0])

    def initialize(self, rng: PRNGKey):
        dkey, akey = jax.random.split(rng)
        self.decision_state = create_state(dkey, CPM, self.dmc, self.doc,
                                           [(1, self.dmc.condition_dim),
                                            (1, self.dmc.decision_dim)])  # type: TrainState

        self.adversary_state = create_state(akey, RatioModel, self.amc, self.aoc,
                                            [(1, self.amc.features[0])])  # type: TrainState

        @jax.jit
        def _update_step(key: PRNGKey, decision_state: TrainState, adversary_state: TrainState,
                         lagrange_multiplier: jnp.ndarray, batch: jnp.ndarray):
            def _inner_loss_fn(decision_params, adversary_params, lagrange_param, update_stats=False):
                log_pi_theta = decision_state.apply_fn({'params': decision_params},
                                                       condition=batch[:, self.gc.batch_condition_dims],
                                                       decision=batch[:, self.gc.batch_decision_dims])

                # args: conditions, decision
                log_p_true = self.log_p_true(batch[:, self.gc.batch_condition_dims],
                                             batch[:, self.gc.batch_decision_dims])

                r_phi, adversary_mutable = adversary_state.apply_fn(
                    {'params': adversary_params, **adversary_state.state},
                    batch,
                    mutable=list(adversary_state.state.keys()),
                    train=True,
                    update_stats=update_stats)

                kl = jnp.mean((jnp.log(r_phi) - log_pi_theta + log_p_true) * r_phi)
                target_loss = jnp.mean(r_phi * self.loss(batch))
                loss = -target_loss + lagrange_param * (kl - self.gc.alpha)
                output_dict = {'kl': kl, 'target_loss': target_loss}
                return loss, adversary_mutable, output_dict

            def _gradient_descent(decision_params):
                def _inner_fn(i: int, params: Tuple):
                    """params: (adversary_state, lagrange_params)"""
                    adv_state, lagrange_params = params

                    (adv_grads, lag_grads) = jax.grad(_inner_loss_fn, has_aux=True, argnums=(1, 2))(decision_params,
                                                                                                    adv_state.params,
                                                                                                    lagrange_params)
                    # Do not update the batch stats during the inner loop
                    new_adverse_state = adv_state.apply_gradients(grads=adv_grads)

                    # Ensure non-negative lambda and lambda <= max_lambda
                    new_lagrange_params = lagrange_params - self.gc.lagrange_lr * lag_grads
                    new_lagrange_params = jnp.maximum(new_lagrange_params, jnp.array([0.0]))
                    new_lagrange_params = jnp.minimum(new_lagrange_params, jnp.array([self.gc.max_lambda]))

                    return new_adverse_state, new_lagrange_params

                new_adversary_state, new_lagrange_multiplier = jax.lax.fori_loop(0, self.gc.inner_steps, _inner_fn,
                                                                                 (adversary_state, lagrange_multiplier))

                r_phi, adv_mutable = adversary_state.apply_fn(
                    {'params': new_adversary_state.params, **adversary_state.state},
                    batch,
                    mutable=list(new_adversary_state.state.keys()),
                    train=True)

                new_adversary_state.replace(state=adv_mutable)  # Update the batch stats in the last call
                outer_loss = jnp.mean(r_phi * self.loss(batch))
                return outer_loss, new_adversary_state, new_lagrange_multiplier

            (_, new_adv_state, new_lag_multiplier), grads = (
                jax.value_and_grad(_gradient_descent, has_aux=True)(decision_state.params)
            )

            new_decision_state = decision_state.apply_gradients(grads=grads)
            _, _, outputs = _inner_loss_fn(new_decision_state.params, new_adv_state.params, new_lag_multiplier)
            outputs['lagrange_multiplier'] = new_lag_multiplier

            return new_decision_state, new_adv_state, new_lag_multiplier, outputs

        @jax.jit
        def _eval_fn(key: PRNGKey, decision_state: TrainState, adversary_state: TrainState, batch: jnp.ndarray):
            log_pi_theta = decision_state.apply_fn({'params': decision_state.params},
                                                   condition=batch[:, self.gc.batch_condition_dims],
                                                   decision=batch[:, self.gc.batch_decision_dims],
                                                   train=False)
            r_phi = adversary_state.apply_fn({'params': adversary_state.params, **adversary_state.state},
                                             batch,
                                             train=False)
            log_p_true = self.log_p_true(batch[:, self.gc.batch_condition_dims], batch[:, self.gc.batch_decision_dims])
            kl = jnp.mean((jnp.log(r_phi) - log_pi_theta + log_p_true) * r_phi)
            target_loss = jnp.mean(r_phi * self.loss(batch))

            return {'kl': kl, 'target_loss': target_loss}

        self._update_step = _update_step
        self._eval_fn = _eval_fn

    def train_step(self, key: PRNGKey, batch: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        self.decision_state, self.adversary_state, self.lagrange_multiplier, outputs = self._update_step(
            key, self.decision_state, self.adversary_state, self.lagrange_multiplier, batch
        )
        return outputs

    def eval_step(self, key: PRNGKey, batch: jnp.ndarray):
        return self._eval_fn(key, self.decision_state, self.adversary_state, batch)
