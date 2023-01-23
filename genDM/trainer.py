# Copyright (c) 2022, Vahid Balazadeh Meresht
# MIT License

from typing import Callable, Tuple
from util.commons import PRNGKey

import jax
import jax.numpy as jnp

from util.utils import ConfigDict, TrainState, create_state
from genDM.models import RatioModel, CPM
from genDM.hypergradients import hypergradient


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

        self.decision_state, self.base_adversary_state = None, None
        self._update_step = None
        self._eval_fn = None

        self.train_metrics = {}
        self.eval_metrics = {}

        self.base_lagrange_multiplier = jnp.array(0., dtype=jnp.float32)

    def initialize(self, rng: PRNGKey):
        dkey, akey = jax.random.split(rng)
        self.decision_state = create_state(dkey, CPM, self.dmc, self.doc,
                                           [(1, self.dmc.condition_dim),
                                            (1, self.dmc.decision_dim)])  # type: TrainState

        self.base_adversary_state = create_state(akey, RatioModel, self.amc, self.aoc,
                                                 [(1, self.amc.input_dim)])  # type: TrainState

        def inner_loss_fn(decision_params, adversary_params, adversary_stats, lagrange_param, batch,
                          update_stats):
            log_pi_theta = self.decision_state.apply_fn({'params': decision_params},
                                                        conditions=batch[:, self.gc.batch_condition_dims],
                                                        decision=batch[:, self.gc.batch_decision_dims])

            log_p_true = self.log_p_true(batch[:, self.gc.batch_condition_dims],
                                         batch[:, self.gc.batch_decision_dims])  # args: conditions, decision

            r, adversary_mutable = self.base_adversary_state.apply_fn({'params': adversary_params, **adversary_stats},
                                                                      batch,
                                                                      mutable=list(adversary_stats.keys()),
                                                                      train=True,
                                                                      update_stats=update_stats)
            kl = jnp.mean(jnp.log(r) * r) - jnp.mean((log_pi_theta - log_p_true) * r)
            target_loss = jnp.mean(r * self.loss(batch))
            loss = - target_loss + lagrange_param * (kl - self.gc.alpha)
            output_dict = {'kl': kl, 'target_loss': target_loss, 'lagrange': lagrange_param}
            debug_dict = {'avg_log_policy': jnp.mean(log_pi_theta),
                          'avg_loss': jnp.mean(self.loss(batch))}
            return loss, (adversary_mutable, output_dict, debug_dict)

        def outer_loss_fn(adversary_params, adversary_stats, batch, update_stats):
            r, adversary_mutable = self.base_adversary_state.apply_fn({'params': adversary_params, **adversary_stats},
                                                                      batch,
                                                                      mutable=list(adversary_stats.keys()),
                                                                      train=True,
                                                                      update_stats=update_stats)
            return jnp.mean(r * self.loss(batch)), (adversary_mutable,)

        @jax.jit
        def _update_step(key: PRNGKey, decision_state: TrainState, base_adversary_state: TrainState,
                         base_lagrange_multiplier: jnp.ndarray, batch: jnp.ndarray):
            def _inner_update_step(_, params):
                """params: (adversary_state, lagrange_params)"""
                adversary_state, lagrange_params = params

                (
                    (adv_grads, lag_grads), (adv_mutable, _, _)
                ) = jax.grad(inner_loss_fn, has_aux=True, argnums=(1, 3))(decision_state.params,
                                                                          adversary_state.params,
                                                                          adversary_state.stats,
                                                                          lagrange_params,
                                                                          batch,
                                                                          update_stats=True)
                # Update the batch stats during the inner loop
                new_adverse_state = adversary_state.apply_gradients(grads=adv_grads, stats=adv_mutable)

                # Gradient ascent on the lagrange multiplier + Ensure non-negative lambda and lambda <= max_lambda
                new_lagrange_params = lagrange_params + self.gc.lagrange_lr * lag_grads

                new_lagrange_params = jnp.maximum(new_lagrange_params, 0)
                new_lagrange_params = jnp.minimum(new_lagrange_params, self.gc.max_lambda)

                return new_adverse_state, new_lagrange_params

            new_adversary_state, new_lagrange_multiplier = jax.lax.fori_loop(0, self.gc.inner_steps,
                                                                             _inner_update_step,
                                                                             (base_adversary_state,
                                                                              base_lagrange_multiplier))

            _inner_loss_fn = lambda adv_params, dec_params: inner_loss_fn(dec_params, adv_params,
                                                                          new_adversary_state.stats,
                                                                          new_lagrange_multiplier,
                                                                          batch, update_stats=False)[0]

            _outer_loss_fn = lambda adv_params: outer_loss_fn(adv_params, new_adversary_state.stats,
                                                              batch, update_stats=False)[0]
            decision_grads = hypergradient(val_loss=_outer_loss_fn, train_loss=_inner_loss_fn,
                                           params=new_adversary_state.params, hyperparams=decision_state.params,
                                           inverse_steps=self.gc.neumann_steps, lr=self.gc.neumann_lr)

            new_decision_state = decision_state.apply_gradients(grads=decision_grads)

            # Output the log for the current decision state and the new adversary state
            _, (_, outputs, debugs) = inner_loss_fn(decision_state.params, new_adversary_state.params,
                                                    adversary_stats=new_adversary_state.stats,
                                                    lagrange_param=new_lagrange_multiplier,
                                                    batch=batch, update_stats=False)

            # decision_grads = jax.tree_map(lambda x: jnp.mean(x), decision_grads)
            # debugs['theta_grads'] = decision_grads
            debugs['mean_r_raw'] = jnp.mean(new_adversary_state.stats['batch_stats']['mean'])

            return new_decision_state, outputs, debugs

        # @jax.jit
        # def _eval_fn(key: PRNGKey, decision_state: TrainState, adversary_state: TrainState, batch: jnp.ndarray):
        #     log_pi_theta = decision_state.apply_fn({'params': decision_state.params},
        #                                            conditions=batch[:, self.gc.batch_condition_dims],
        #                                            decision=batch[:, self.gc.batch_decision_dims])
        #     r_phi = adversary_state.apply_fn({'params': adversary_state.params, **adversary_state.state},
        #                                      batch,
        #                                      train=False)
        #     log_p_true = self.log_p_true(batch[:, self.gc.batch_condition_dims],
        #                                  batch[:, self.gc.batch_decision_dims])
        #     kl = jnp.mean((jnp.log(r_phi) - log_pi_theta + log_p_true) * r_phi)
        #     target_loss = jnp.mean(r_phi * self.loss(batch))
        #     return {'kl': kl, 'target_loss': target_loss}

        self._update_step = _update_step
        # self._eval_fn = _eval_fn

    def train_step(self, key: PRNGKey, batch: jnp.ndarray) -> Tuple[dict, ...]:
        self.decision_state, outputs, debug = self._update_step(key,
                                                                self.decision_state,
                                                                self.base_adversary_state,
                                                                self.base_lagrange_multiplier,
                                                                batch)
        # print(debug.pop('theta_grads'))
        return outputs, debug

    # def eval_step(self, key: PRNGKey, batch: jnp.ndarray):
    #     return self._eval_fn(key, self.decision_state, self.adversary_state, batch)


class OptimalAdvTrainer:
    """Training the generative Causal Decision-Making. It solves the following optimization:
    min_{theta} min_{lambda >= 0}
        lambda . log E_{v ~ p(V)} [exp(loss(v) / lambda) . pi_{theta}(t|x) / p_true(t|x)] + lambda alpha
    """

    def __init__(self,
                 target_loss: Callable[[jnp.ndarray], jnp.ndarray],
                 log_p_true: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],  # args: conditions, decision
                 global_config: ConfigDict,
                 decision_model_config: ConfigDict, decision_optim_config: ConfigDict):
        # Has keys: 'batch_condition_dims', 'batch_decision_dims', 'lagrange_lr', 'alpha', 'max_lambda', 'inner_steps'
        self.gc = global_config
        self.dmc = decision_model_config
        self.doc = decision_optim_config

        self.log_p_true = log_p_true

        self.loss = target_loss

        self.decision_state = None
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

        def loss_fn(decision_params, lagrange_param, batch):
            log_pi_theta = self.decision_state.apply_fn({'params': decision_params},
                                                        conditions=batch[:, self.gc.batch_condition_dims],
                                                        decision=batch[:, self.gc.batch_decision_dims])

            log_p_true = self.log_p_true(batch[:, self.gc.batch_condition_dims],
                                         batch[:, self.gc.batch_decision_dims])  # args: conditions, decision
            loss = (
                    lagrange_param
                    * jnp.log(jnp.exp(self.loss(batch) / lagrange_param) * jnp.exp(log_pi_theta - log_p_true)).mean()
                    + lagrange_param * self.gc.alpha
            )
            return loss

        @jax.jit
        def _update_step(key: PRNGKey, decision_state: TrainState, lagrange_mult: jnp.ndarray, batch: jnp.ndarray):
            loss, (decision_grads, lagrange_grads) = jax.value_and_grad(loss_fn, argnums=(0, 1))(decision_state.params,
                                                                                                 lagrange_mult, batch)

            new_lagrange_params = lagrange_mult - self.gc.lagrange_lr * lagrange_grads

            new_lagrange_params = jnp.maximum(new_lagrange_params, 0)
            new_lagrange_params = jnp.minimum(new_lagrange_params, self.gc.max_lambda)

            new_decision_state = decision_state.apply_gradients(grads=decision_grads)

            outputs = {'loss': loss, 'lagrange_mult': lagrange_mult}
            return new_decision_state, outputs

        self._update_step = _update_step

    def train_step(self, key: PRNGKey, batch: jnp.ndarray) -> Tuple[dict, ...]:
        self.decision_state, outputs, _ = self._update_step(key,
                                                            self.decision_state,
                                                            self.lagrange_multiplier,
                                                            batch)
        return outputs, {}


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

        def inner_fn(_, x):
            return x - 0.01 * jax.grad(inner_loss)(x)

        x_new = jax.lax.fori_loop(0, 500, inner_fn, 1.0)
        return x_new * y

    def outer_fn(_, y):
        return y - 0.01 * jax.grad(gradient_descent)(y)

    assert jnp.isclose(jax.lax.fori_loop(0, 500, outer_fn, 1.0), 0.0, atol=1e-4)


def test_MLE():
    import numpy as np
    from util.utils import PRNGSequence
    import tqdm
    import matplotlib.pyplot as plt

    gc = ConfigDict(
        {
            'batch_condition_dims': np.array([0]),
            'batch_decision_dims': np.array([1]),
            'epochs': 1000,
        }
    )
    dmc = ConfigDict(
        {
            'features': [32, 32, 1],  # without input/output
            'activation': 'tanh',
            'condition_dim': 1,
            'decision_dim': 1,
            'variance': None
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
    data_x = jax.random.normal(next(rng), (1000, 1))
    data_t = jax.random.normal(next(rng), (1000, 1)) + 2 * data_x
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

    mu_x = mle_trainer.decision_state.apply_fn({'params': mle_trainer.decision_state.params}, data_x, method=CPM.mu)
    plt.plot(data_x, mu_x, 'o')
    plt.savefig('outputs/mle.png')
    plt.close()
