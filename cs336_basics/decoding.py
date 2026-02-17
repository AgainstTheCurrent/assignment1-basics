"""
JAX text generation with KV-cache optimized autoregressive decoding.

Original PyTorch version preserved at cs336_basics/pytorch/decoding.py.

Design
------
During generation each step processes only the *new* token(s) through the
model and reuses the cached K/V projections from all previous steps.  This
reduces per-step cost from O(seq_len * d^2) to O(d^2) for the attention
projection and from O(seq_len^2 * d) to O(seq_len * d) for the dot-product
attention.

The cache uses preallocated fixed-size buffers (max_seq_len) so that every
call has a static pytree structure and array shapes, enabling JAX JIT.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from cs336_basics.modules import TransformerLM, ModelCache
from cs336_basics.nn_utils import softmax


# ---------------------------------------------------------------------------
# JIT-compiled single decode step
# ---------------------------------------------------------------------------

@nnx.jit
def _decode_step(
    model: TransformerLM,
    token_ids: jnp.ndarray,
    positions: jnp.ndarray,
    cache: ModelCache,
    rng: jax.Array,
    temperature: jnp.ndarray,
    top_p: jnp.ndarray,
) -> tuple[jnp.ndarray, ModelCache, jax.Array]:
    """Single jit-compiled decode step: forward + nucleus sample.

    Returns (next_token, new_cache, next_rng).
    """
    logits, new_cache = model.forward_with_cache(token_ids, positions, cache)
    next_logits = logits[0, -1, :]  # (vocab_size,)

    # Temperature scaling + nucleus sampling
    scaled = next_logits / temperature
    probs = softmax(scaled, axis=-1)
    sorted_indices = jnp.argsort(-probs)
    sorted_probs = probs[sorted_indices]
    cumulative = jnp.cumsum(sorted_probs)
    cutoff = jnp.searchsorted(cumulative, top_p)
    mask = jnp.arange(probs.shape[0]) <= cutoff
    filtered = jnp.where(mask, sorted_probs, 0.0)
    filtered = filtered / jnp.sum(filtered)

    rng, sample_rng = jax.random.split(rng)
    sampled_idx = jax.random.categorical(sample_rng, jnp.log(filtered + 1e-30))
    next_token = sorted_indices[sampled_idx]

    return next_token, new_cache, rng


# ---------------------------------------------------------------------------
# Top-p (nucleus) sampling
# ---------------------------------------------------------------------------

def generate_text(
    model: TransformerLM,
    input_ids: jnp.ndarray,
    max_new_tokens: int,
    p: float,
    temperature: float = 1.0,
    stop_token_ids: Optional[list[int]] = None,
    *,
    rng: jax.Array,
):
    """Generate tokens using nucleus (top-p) sampling with KV-cache.

    Yields one token ID (int) at a time for streaming output.
    Each decode step is JIT-compiled.  The KV cache uses preallocated
    fixed-size buffers so that array shapes are static across steps.

    Args:
        model: A ``TransformerLM`` instance (Flax NNX).
        input_ids: 1-D int array of prompt token IDs *(seq_len,)*.
        max_new_tokens: Maximum number of new tokens to generate.
        p: Cumulative probability threshold for nucleus sampling.
        temperature: Temperature scaling (higher = more random).
        stop_token_ids: Optional list of IDs that terminate generation.
        rng: JAX PRNG key used for sampling.

    Yields:
        int — one token ID per step.
    """
    prompt_len = input_ids.shape[0]
    batch_ids = input_ids[None, :]                      # (1, seq_len)
    positions = jnp.arange(prompt_len)[None, :]         # (1, seq_len)

    # Preallocate fixed-size cache
    cache = model.init_cache(batch_size=1)

    temp_arr = jnp.array(temperature, dtype=jnp.float32)
    p_arr = jnp.array(p, dtype=jnp.float32)

    # Prefill (JIT-compiled)
    next_token, cache, rng = _decode_step(
        model, batch_ids, positions, cache, rng, temp_arr, p_arr)

    token_id = int(next_token)
    if stop_token_ids and token_id in stop_token_ids:
        return
    yield token_id
    cur_pos = prompt_len

    # Autoregressive decode
    for _ in range(max_new_tokens - 1):
        new_ids = jnp.array([[token_id]], dtype=jnp.int32)     # (1, 1)
        new_pos = jnp.array([[cur_pos]], dtype=jnp.int32)      # (1, 1)

        next_token, cache, rng = _decode_step(
            model, new_ids, new_pos, cache, rng, temp_arr, p_arr)

        token_id = int(next_token)
        if stop_token_ids and token_id in stop_token_ids:
            return
        yield token_id
        cur_pos += 1
