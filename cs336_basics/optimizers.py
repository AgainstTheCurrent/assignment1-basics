"""
JAX / optax implementation of optimizers.

Original PyTorch version preserved at cs336_basics/pytorch/optimizers.py.
"""

import jax.numpy as jnp
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


# ---------------------------------------------------------------------------
# SGD with decaying learning rate: lr / sqrt(t + 1)
# ---------------------------------------------------------------------------

class SGDState(NamedTuple):
    """State for the decaying-LR SGD optimizer."""
    t: jnp.ndarray  # scalar int32, iteration counter


def sgd(lr: float = 1e-3) -> optax.GradientTransformation:
    """SGD with learning rate decayed by 1/sqrt(t+1) at each step."""

    def init_fn(params):
        del params
        return SGDState(t=jnp.zeros([], jnp.int32))

    def update_fn(updates, state, params=None):
        del params
        t = state.t
        scale = -lr / jnp.sqrt(t.astype(jnp.float32) + 1.0)
        new_updates = jax.tree.map(lambda g: scale * g, updates)
        return new_updates, SGDState(t=t + 1)

    return optax.GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# AdamW (decoupled weight decay, bias-corrected moments)
# ---------------------------------------------------------------------------

class AdamWState(NamedTuple):
    """State for the AdamW optimizer."""
    t: jnp.ndarray    # scalar int32, starts at 1 after first step
    m: Any               # first moment pytree (same structure as params)
    v: Any               # second moment pytree (same structure as params)


def adamw(
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    """AdamW optimizer matching the behaviour of the PyTorch reference.

    Bias-corrected first/second moments with decoupled weight decay applied
    *after* the Adam update (i.e. ``p -= lr * wd * p`` each step).
    """
    beta1, beta2 = betas

    def init_fn(params):
        m = jax.tree.map(jnp.zeros_like, params)
        v = jax.tree.map(jnp.zeros_like, params)
        return AdamWState(t=jnp.ones([], jnp.int32), m=m, v=v)

    def update_fn(updates, state, params=None):
        t = state.t

        # Update biased first & second moments
        new_m = jax.tree.map(
            lambda mi, gi: beta1 * mi + (1 - beta1) * gi, state.m, updates)
        new_v = jax.tree.map(
            lambda vi, gi: beta2 * vi + (1 - beta2) * gi ** 2, state.v, updates)

        # Bias-corrected learning rate
        lr_t = lr * jnp.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

        # Adam update: -lr_t * m / (sqrt(v) + eps)
        adam_updates = jax.tree.map(
            lambda mi, vi: -lr_t * mi / (jnp.sqrt(vi) + eps), new_m, new_v)

        # Decoupled weight decay: -lr * wd * p
        if params is not None:
            new_updates = jax.tree.map(
                lambda au, p: au - lr * weight_decay * p, adam_updates, params)
        else:
            new_updates = adam_updates

        return new_updates, AdamWState(t=t + 1, m=new_m, v=new_v)

    return optax.GradientTransformation(init_fn, update_fn)


# ---------------------------------------------------------------------------
# Cosine learning rate schedule with linear warmup (pure math, no JAX needed)
# ---------------------------------------------------------------------------

def get_lr_cosine_schedule(
    it,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Compute the learning rate at a given iteration using a cosine schedule
    with linear warmup.

    Uses jnp ops so the function is JAX-traceable (works inside jit).

    Args:
        it: The current iteration number (Python int or JAX scalar).
        max_learning_rate: The maximum learning rate.
        min_learning_rate: The minimum learning rate.
        warmup_iters: Number of iterations for linear warmup.
        cosine_cycle_iters: Total iterations for one cosine cycle.

    Returns:
        The learning rate for iteration *it*.
    """
    it = jnp.asarray(it, dtype=float)
    warmup_lr = max_learning_rate * it / warmup_iters
    cosine_decay = min_learning_rate + 0.5 * (
        1 + jnp.cos(
            (it - warmup_iters)
            / (cosine_cycle_iters - warmup_iters)
            * jnp.pi
        )
    ) * (max_learning_rate - min_learning_rate)
    return jnp.where(
        it < warmup_iters,
        warmup_lr,
        jnp.where(it <= cosine_cycle_iters, cosine_decay, min_learning_rate),
    )
