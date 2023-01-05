from util.commons import Params, Any
from typing import Callable

import jax

from functools import partial


# Adapted from Jonathan Lorraine, et al. https://arxiv.org/pdf/1911.02590.pdf (Algorithm 2, 3)
@partial(jax.jit, static_argnums=(0, 1, 4))
def hypergradient(val_loss: Callable[[Params], Any], train_loss: Callable[[Params, Params], Any],
                  params: Params, hyperparams: Params, inverse_steps: int, lr: float):
    """
    Compute the hypergradient of the validation loss with respect to the hyperparameters at the
    optimal parameters.

    Args:
        val_loss: A function that takes the parameters and returns the validation loss.
        train_loss: A function that takes the parameters (arg num 0) and hyperparameters (arg num 1).
        params: The (near)-optimal parameters.
        hyperparams: The current hyperparameters.
        inverse_steps: The number of steps to calculate the inverse Hessian.
        lr: The learning rate for the inverse Hessian calculation.
    """

    def approx_inverse_HVP(v, f):
        """Approximate the inverse Hessian-vector product v [df/dw]^{-1}."""
        p = v
        for j in range(inverse_steps):
            v -= lr * jax.vjp(f, params)[1](v)[0]
            p += v
        return p

    v1 = jax.grad(val_loss)(params)
    d_train_d_w = lambda w: jax.grad(train_loss, argnums=0)(w, hyperparams)
    v2 = approx_inverse_HVP(v1, d_train_d_w)  # [dval/dw] · [d^2train/dw^2]^{-1}
    d_train_d_hyper = lambda w: jax.grad(train_loss, argnums=1)(w, hyperparams)
    v3 = jax.vjp(d_train_d_hyper, params)[1](v2)[0]  # [dval/dw] · [d^2train/dw^2]^{-1} · [d^2train/dw dλ]

    return - v3


def test_hyper_gradient():
    import jax.numpy as jnp
    import tqdm

    @jax.jit
    def update_param(p, hp):
        def inner_loss(param, hyperparam):
            return (param - jnp.exp(hyperparam)) ** 2

        new_p = p - 0.01 * jax.grad(inner_loss, argnums=0)(p, hp)
        return new_p

    @jax.jit
    def get_hyper_grads(p, hp):
        def inner_loss(param, hyperparam):
            return (param - jnp.exp(hyperparam)) ** 2

        def outer_loss(param):
            return (param - 1) ** 2

        return hp - 0.01 * hypergradient(outer_loss, inner_loss, p, hp, 10, 0.5)

    init_hyper_param = -1.
    init_param = 0.
    with tqdm.tqdm(total=500) as pbar:
        for i in range(500):  # Update hyperparams
            for j in range(50):  # Find the optimal params
                init_param = update_param(init_param, init_hyper_param)
            init_hyper_param = get_hyper_grads(init_param, init_hyper_param)
            pbar.set_postfix_str(f"Hyperparam: {init_hyper_param}, Param: {init_param}")
            pbar.update(1)

    assert jnp.isclose(init_param, 1., atol=1e-4), "The optimal param should be 1."

