"""
JAX implementation of neural network utility functions.

Original PyTorch version preserved at cs336_basics/pytorch/nn_utils.py.
"""

from typing import Optional

import jax
import jax.numpy as jnp


def softmax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    x_max = jnp.max(x, axis=axis, keepdims=True)
    e_x = jnp.exp(x - x_max)
    return e_x / jnp.sum(e_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: jnp.ndarray,
    K: jnp.ndarray,
    V: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Scaled dot-product attention.

    Args:
        Q: (..., queries, d_k)
        K: (..., keys,   d_k)
        V: (..., keys,   d_v)
        mask: (..., queries, keys) -- True = attend, False = masked

    Returns:
        (..., queries, d_v)
    """
    d_k = Q.shape[-1]
    scores = jnp.einsum('...nk,...mk->...nm', Q, K) / jnp.sqrt(jnp.array(d_k, dtype=Q.dtype))
    if mask is not None:
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    attn_weights = softmax(scores, axis=-1)
    return jnp.einsum('...nm,...mk->...nk', attn_weights, V)


def cross_entropy(
    inputs: jnp.ndarray,
    targets: jnp.ndarray,
) -> jnp.ndarray:
    """Average cross-entropy loss.

    Args:
        inputs: (..., seq_len, vocab_size) -- unnormalised logits.
        targets: (..., seq_len) -- integer class indices.

    Returns:
        Scalar mean cross-entropy loss.
    """
    inputs_max = jnp.max(inputs, axis=-1, keepdims=True)
    inputs = inputs - inputs_max  # numerical stability
    log_sum_exp = jnp.log(jnp.sum(jnp.exp(inputs), axis=-1))
    target_logits = jnp.take_along_axis(inputs, targets[..., None].astype(jnp.int32), axis=-1).squeeze(-1)
    return jnp.mean(log_sum_exp - target_logits)


def gradient_clipping(
    params: dict,
    max_l2_norm: float,
) -> dict:
    """Clip parameter gradients by global L2 norm.

    In JAX, gradients are just pytrees (dicts of arrays), not mutable
    Parameter objects.  This function takes a gradient pytree and returns
    a clipped copy.

    Args:
        params: A pytree of gradient arrays (same structure as model params).
        max_l2_norm: Maximum allowed L2 norm.

    Returns:
        Clipped gradient pytree with the same structure.
    """
    leaves = jax.tree.leaves(params)
    total_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    clip_coef = jnp.minimum(clip_coef, 1.0)
    return jax.tree.map(lambda g: g * clip_coef, params)
