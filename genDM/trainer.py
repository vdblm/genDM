# Copyright (c) 2022, Vahid Balazadeh Meresht
# MIT License

from typing import Callable, Tuple
from util.commons import PRNGKey

import jax
import jax.numpy as jnp

from util.utils import ConfigDict, TrainState, create_state
from genDM.models import RatioModel, CPM


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

        self.train_metrics = {}
        self.eval_metrics = {}

        self.lagrange_multiplier = jnp.array(0., dtype=jnp.float32)

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
            @jax.jit
            def _inner_loss_fn(decision_params, adversary_params, lagrange_param, update_stats=False):
                log_pi_theta = decision_state.apply_fn({'params': decision_params},
                                                       conditions=batch[:, self.gc.batch_condition_dims],
                                                       decision=batch[:, self.gc.batch_decision_dims])

                log_p_true = self.log_p_true(batch[:, self.gc.batch_condition_dims],
                                             batch[:, self.gc.batch_decision_dims])  # args: conditions, decision

                r, adversary_mutable = adversary_state.apply_fn(
                    {'params': adversary_params, **adversary_state.state},
                    batch,
                    mutable=list(adversary_state.state.keys()),
                    train=True,
                    update_stats=update_stats)
                weighted_policy_ratio = jnp.mean((log_pi_theta - log_p_true) * r)
                kl = jnp.mean(jnp.log(r) * r) - weighted_policy_ratio
                target_loss = jnp.mean(r * self.loss(batch))
                loss = - target_loss + lagrange_param * (kl - self.gc.alpha)
                output_dict = {'kl': kl, 'target_loss': target_loss, 'lagrange': lagrange_param}
                debug_dict = {'weighted_policy_ratio': weighted_policy_ratio,
                              'avg_loss': jnp.mean(self.loss(batch))}
                return loss, (adversary_mutable, output_dict, debug_dict)

            @jax.jit
            def _gradient_descent(decision_params):
                def _inner_fn(i: int, params: Tuple):
                    """params: (adversary_state, lagrange_params)"""
                    adv_state, lagrange_params = params

                    (
                        (adv_grads, lag_grads), _
                    ) = jax.grad(_inner_loss_fn, has_aux=True, argnums=(1, 2))(decision_params,
                                                                               adv_state.params,
                                                                               lagrange_params)
                    # Do not update the batch stats during the inner loop
                    new_adverse_state = adv_state.apply_gradients(grads=adv_grads)

                    # Gradient ascent on the lagrange multiplier
                    new_lagrange_params = lagrange_params + self.gc.lagrange_lr * lag_grads
                    # Ensure non-negative lambda and lambda <= max_lambda
                    new_lagrange_params = jnp.maximum(new_lagrange_params, 0)
                    new_lagrange_params = jnp.minimum(new_lagrange_params, self.gc.max_lambda)

                    return new_adverse_state, new_lagrange_params

                new_adversary_state, new_lagrange_multiplier = jax.lax.fori_loop(0, self.gc.inner_steps, _inner_fn,
                                                                                 (adversary_state, lagrange_multiplier))

                r_phi_new, adv_mutable = new_adversary_state.apply_fn(
                    {'params': new_adversary_state.params, **new_adversary_state.state},
                    batch,
                    mutable=list(new_adversary_state.state.keys()),
                    train=True,
                    update_stats=True)

                # Update the batch stats in the last call
                new_adversary_state = new_adversary_state.replace(state=adv_mutable)
                outer_loss = jnp.mean(r_phi_new * self.loss(batch))
                return outer_loss, (r_phi_new, new_adversary_state, new_lagrange_multiplier)

            (_, (r_phi, new_adv_state, new_lag_multiplier)), decision_grads = (
                jax.value_and_grad(_gradient_descent, has_aux=True)(decision_state.params)
            )
            new_decision_state = decision_state.apply_gradients(grads=decision_grads)
            _, (_, outputs, debugs) = _inner_loss_fn(new_decision_state.params, new_adv_state.params,
                                                     new_lag_multiplier)
            # decision_grads = jax.tree_map(lambda x: jnp.mean(x), decision_grads)
            # debugs['theta_grads'] = decision_grads
            debugs['max_r_phi'] = jnp.max(r_phi)
            debugs['mean_r_raw'] = jnp.mean(new_adv_state.state['batch_stats']['mean'])
            return new_decision_state, new_adv_state, new_lag_multiplier, outputs, debugs

        @jax.jit
        def _eval_fn(key: PRNGKey, decision_state: TrainState, adversary_state: TrainState, batch: jnp.ndarray):
            log_pi_theta = decision_state.apply_fn({'params': decision_state.params},
                                                   conditions=batch[:, self.gc.batch_condition_dims],
                                                   decision=batch[:, self.gc.batch_decision_dims])
            r_phi = adversary_state.apply_fn({'params': adversary_state.params, **adversary_state.state},
                                             batch,
                                             train=False)
            log_p_true = self.log_p_true(batch[:, self.gc.batch_condition_dims],
                                         batch[:, self.gc.batch_decision_dims])
            kl = jnp.mean((jnp.log(r_phi) - log_pi_theta + log_p_true) * r_phi)
            target_loss = jnp.mean(r_phi * self.loss(batch))
            return {'kl': kl, 'target_loss': target_loss}

        self._update_step = _update_step
        self._eval_fn = _eval_fn

    def train_step(self, key: PRNGKey, batch: jnp.ndarray) -> Tuple[dict, ...]:
        self.decision_state, self.adversary_state, self.lagrange_multiplier, outputs, debug = self._update_step(
            key, self.decision_state, self.adversary_state, self.lagrange_multiplier, batch
        )
        return outputs, debug

    def eval_step(self, key: PRNGKey, batch: jnp.ndarray):
        return self._eval_fn(key, self.decision_state, self.adversary_state, batch)


class MLETrainer:
    """Trainer for maximum likelihood estimation of p_true(t|x)."""

    def __init__(self, global_config: ConfigDict, decision_model_config: ConfigDict, decision_optim_config: ConfigDict):
        self.gc = global_config
        self.dmc = decision_model_config
        self.doc = decision_optim_config

        self.decision_state = None
        self._update_step = None

    def initialize(self, rng: PRNGKey):
        k, _ = jax.random.split(rng)
        self.decision_state = create_state(k, CPM, self.dmc, self.doc,
                                           [(1, self.dmc.condition_dim),
                                            (1, self.dmc.decision_dim)])  # type: TrainState

        @jax.jit
        def _update_step(key: PRNGKey, decision_state: TrainState, batch: jnp.ndarray):
            def _loss_fn(params):
                log_p = decision_state.apply_fn({'params': params},
                                                conditions=batch[:, self.gc.batch_condition_dims],
                                                decision=batch[:, self.gc.batch_decision_dims])

                return -jnp.mean(log_p)

            loss, grads = jax.value_and_grad(_loss_fn)(decision_state.params)
            new_decision_state = decision_state.apply_gradients(grads=grads)
            return new_decision_state, {'log_likelihood': -loss}

        self._update_step = _update_step

    def train_step(self, key: PRNGKey, batch: jnp.ndarray) -> dict:
        self.decision_state, outputs = self._update_step(key, self.decision_state, batch)
        return outputs

    def log_p(self, conditions: jnp.ndarray, decision: jnp.ndarray):
        return self.decision_state.apply_fn({'params': self.decision_state.params},
                                            conditions=conditions,
                                            decision=decision)


def test_meta_gradient():
    def gradient_descent(y):
        def inner_loss(x):
            return (x - y) ** 2

        def inner_fn(i, x):
            return x - 0.01 * jax.grad(inner_loss)(x)

        x_new = jax.lax.fori_loop(0, 500, inner_fn, 1.0)
        return x_new * y

    def outer_fn(j, y):
        return y - 0.01 * jax.grad(gradient_descent)(y)

    assert jnp.isclose(jax.lax.fori_loop(0, 500, outer_fn, 1.0), 0.0, atol=1e-4)


def test_MLE():
    import numpy as np
    from util.utils import PRNGSequence
    import tqdm

    gc = ConfigDict(
        {
            'batch_condition_dims': np.array([0]),
            'batch_decision_dims': np.array([1]),
            'epochs': 1000,
        }
    )
    dmc = ConfigDict(
        {
            'features': [1, 32, 32, 1],  # without input/output
            'activation': 'tanh',
            'condition_dim': 1,
            'decision_dim': 1,
            'variance': 1.
        }
    )

    doc = ConfigDict(
        {
            'learning_rate': 0.01,
            'optimizer': 'adam'
        }
    )
    rng = PRNGSequence(38488)
    mle_trainer = MLETrainer(gc, dmc, doc)
    mle_trainer.initialize(next(rng))
    data_x = jax.random.normal(next(rng), (100, 1))
    data_t = jax.random.normal(next(rng), (100, 1)) + 2 * data_x
    data = jnp.concatenate([data_x, data_t], axis=1)
    with tqdm.tqdm(total=gc.epochs) as pbar:
        for epoch in range(gc.epochs):
            outputs = mle_trainer.train_step(next(rng), data)
            logging_str = ' '.join(['{}: {: <12g}'.format(k, v) for (k, v) in outputs.items()])
            pbar.set_postfix_str(logging_str)
            pbar.update(1)

    interval = (jnp.min(data_x) - 10, jnp.max(data_x) + 10)
    p_t_x = jax.vmap(lambda t: jnp.exp(mle_trainer.log_p(data_x, t)))
    assert jnp.isclose(1.,
                       p_t_x(jnp.linspace(interval[0], interval[1], 100)).mean() * (interval[1] - interval[0]),
                       atol=1e-2), "Integral of p(t|x) should be 1."
