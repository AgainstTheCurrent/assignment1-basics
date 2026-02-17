"""
JAX / Flax-NNX implementation of all transformer modules.

Original PyTorch version preserved at cs336_basics/pytorch/modules.py.
"""

import math
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax
import flax.nnx as nnx

from cs336_basics.nn_utils import scaled_dot_product_attention


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

class Linear(nnx.Module):
    """Dense linear layer with truncated-normal (Xavier) initialization."""

    def __init__(self, in_features: int, out_features: int, *,
                 rngs: nnx.Rngs):
        std = math.sqrt(2.0 / (in_features + out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nnx.Param(
            jax.random.truncated_normal(
                rngs.params(), lower=-3.0, upper=3.0,
                shape=(out_features, in_features),
            ) * std
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum('...i,oi->...o', x, self.weight[...])


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class Embedding(nnx.Module):
    """Lookup-table embedding with truncated-normal initialization."""

    def __init__(self, num_embeddings: int, embedding_dim: int, *,
                 rngs: nnx.Rngs):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nnx.Param(
            jax.random.truncated_normal(
                rngs.params(), lower=-3.0, upper=3.0,
                shape=(num_embeddings, embedding_dim),
            )  # std=1.0, same as PyTorch version
        )

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        return self.weight[token_ids]


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nnx.Module):
    """Root-Mean-Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5, *,
                 rngs: nnx.Rngs):
        self.d_model = d_model
        self.eps = eps
        self.weight = nnx.Param(jnp.ones(d_model))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_dtype = x.dtype
        x = x.astype(jnp.float32)
        var = jnp.mean(x ** 2, axis=-1, keepdims=True)
        mul = lax.rsqrt(var + self.eps) * self.weight[...]
        return (x * mul).astype(in_dtype)


# ---------------------------------------------------------------------------
# SiLU (Swish) activation
# ---------------------------------------------------------------------------

class SiLU(nnx.Module):
    def __init__(self):
        pass

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x * jax.nn.sigmoid(x)


# ---------------------------------------------------------------------------
# SwiGLU feed-forward
# ---------------------------------------------------------------------------

class SwiGLU(nnx.Module):
    """SwiGLU feed-forward network (Shazeer 2020)."""

    def __init__(self, d_model: int, d_ff: int = 0, *,
                 rngs: nnx.Rngs):
        self.d_model = d_model
        if d_ff == 0:
            d_ff = 8 * d_model // 3
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, rngs=rngs)
        self.w2 = Linear(d_ff, d_model, rngs=rngs)
        self.w3 = Linear(d_model, d_ff, rngs=rngs)
        self.silu = SiLU()

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

class RoPE(nnx.Module):
    """Pre-computed rotary position embeddings."""

    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        position = jnp.arange(max_seq_len)
        dim = jnp.arange(0, d_k, 2)

        # angle[s, d] = s / theta^(2d/d_k)
        angle = jnp.einsum('s,d->sd',
                           position.astype(jnp.float32),
                           1.0 / theta ** (dim.astype(jnp.float32) / d_k))

        self.sin_cached = jnp.sin(angle)
        self.cos_cached = jnp.cos(angle)

    def __call__(self, x: jnp.ndarray, token_positions: jnp.ndarray) -> jnp.ndarray:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)
        """
        sin = self.sin_cached[token_positions]
        cos = self.cos_cached[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        # Interleave even and odd
        x_rot = jnp.stack((x_rot_even, x_rot_odd), axis=-1).reshape(x.shape)
        return x_rot


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nnx.Module):
    """Batched multi-head self-attention with optional RoPE."""

    def __init__(self, d_model: int, num_heads: int, *,
                 rope: bool = False, theta: float = 0.0,
                 max_seq_len: int = 0, rngs: nnx.Rngs):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        if rope:
            self.rope = RoPE(theta=theta, d_k=self.d_k,
                             max_seq_len=max_seq_len)
        else:
            self.rope = None

        # QKV fused projection: (3 * num_heads * d_k, d_model)
        qkv_std = math.sqrt(2.0 / (d_model + self.d_k))
        self.weight_qkv = nnx.Param(
            jax.random.truncated_normal(
                rngs.params(), lower=-3.0, upper=3.0,
                shape=(num_heads * 3, self.d_k, d_model),
            ) * qkv_std
        )
        # Output projection: (d_model, num_heads * d_k)
        o_std = math.sqrt(2.0 / (d_model + d_model))
        self.weight_o = nnx.Param(
            jax.random.truncated_normal(
                rngs.params(), lower=-3.0, upper=3.0,
                shape=(d_model, num_heads, self.d_k),
            ) * o_std
        )

    # -- KV-cache type alias (per layer) ---------------------------------
    # Cache = tuple[jnp.ndarray, jnp.ndarray]   # (K_cache, V_cache)
    #   each of shape (..., num_heads, cached_len, d_k)

    def _project_qkv(self, x: jnp.ndarray):
        """Project x -> Q, K, V each (..., num_heads, seq_len, d_k)."""
        Wx = jnp.einsum('...sd,hkd->...hsk', x, self.weight_qkv[...])
        return jnp.split(Wx, 3, axis=-3)

    def _output_proj(self, attn_out: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum('...hsk,ohk->...so', attn_out, self.weight_o[...])

    def __call__(self, x: jnp.ndarray,
                 token_positions: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """x: (..., seq_len, d_model)  — full-sequence forward (no cache)."""
        seq_len = x.shape[-2]
        Q, K, V = self._project_qkv(x)

        if self.rope is not None:
            if token_positions is None:
                token_positions = jnp.arange(seq_len)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        return self._output_proj(attn_out)

    def forward_with_cache(
        self,
        x: jnp.ndarray,
        token_positions: jnp.ndarray,
        cache: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Cache-aware forward with fixed-size preallocated KV buffers.

        Args:
            x: (batch, new_len, d_model) — typically new_len == 1 during decode.
            token_positions: (batch, new_len) absolute positions for RoPE.
            cache: (K_buf, V_buf, cache_len) where K_buf/V_buf are
                   (batch, num_heads, max_seq_len, d_k) and cache_len is a
                   scalar int32 tracking the number of filled positions.

        Returns:
            (output, new_cache) with new_cache of the same structure.
        """
        Q, K_new, V_new = self._project_qkv(x)

        if self.rope is not None:
            Q = self.rope(Q, token_positions)
            K_new = self.rope(K_new, token_positions)

        K_buf, V_buf, cache_len = cache
        new_len = Q.shape[-2]

        # Write new K/V into the preallocated buffer at cache_len
        start = [0] * (K_buf.ndim - 2) + [cache_len, 0]
        K_buf = jax.lax.dynamic_update_slice(K_buf, K_new, start)
        V_buf = jax.lax.dynamic_update_slice(V_buf, V_new, start)
        new_cache_len = cache_len + new_len

        # Causal + validity mask: query q (abs pos = cache_len + q)
        # can attend to key k if k <= cache_len + q (i.e. k < new_cache_len
        # and causally prior).  Unfilled positions (k >= new_cache_len)
        # are automatically excluded.
        max_len = K_buf.shape[-2]
        key_idx = jnp.arange(max_len)
        query_abs = cache_len + jnp.arange(new_len)
        mask = key_idx[None, :] <= query_abs[:, None]  # (new_len, max_len)

        attn_out = scaled_dot_product_attention(Q, K_buf, V_buf, mask)
        return self._output_proj(attn_out), (K_buf, V_buf, new_cache_len)


# ---------------------------------------------------------------------------
# KV-Cache type
# ---------------------------------------------------------------------------
# Per-layer: (K_buf, V_buf, cache_len) — fixed-size preallocated buffers.
LayerCache = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
ModelCache = list[LayerCache]


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nnx.Module):
    """Pre-norm transformer block: LN -> Attn -> Residual -> LN -> FFN -> Residual."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 theta: float, max_seq_len: int, *, rngs: nnx.Rngs):
        self.attn = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads,
            rope=True, theta=theta, max_seq_len=max_seq_len,
            rngs=rngs)
        self.ln1 = RMSNorm(d_model=d_model, rngs=rngs)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, rngs=rngs)
        self.ln2 = RMSNorm(d_model=d_model, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

    def forward_with_cache(
        self,
        x: jnp.ndarray,
        token_positions: jnp.ndarray,
        cache: LayerCache,
    ) -> tuple[jnp.ndarray, LayerCache]:
        """Cache-aware forward for a single block."""
        attn_out, new_cache = self.attn.forward_with_cache(
            self.ln1(x), token_positions, cache)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_cache


# ---------------------------------------------------------------------------
# Transformer Language Model
# ---------------------------------------------------------------------------

class TransformerLM(nnx.Module):
    """Decoder-only transformer language model."""

    def __init__(self, vocab_size: int, context_length: int, d_model: int,
                 num_layers: int, num_heads: int, d_ff: int, theta: float,
                 *, rngs: nnx.Rngs):
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, rngs=rngs)
        self.layers = nnx.List([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                             theta=theta, max_seq_len=context_length,
                             rngs=rngs)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model=d_model, rngs=rngs)
        self.lm_head = Linear(d_model, vocab_size, rngs=rngs)
        self.context_length = context_length
        self.d_model = d_model
        self.vocab_size = vocab_size

    def __call__(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        """
        token_ids: (batch, seq_len) int array
        Returns: (batch, seq_len, vocab_size) logits
        """
        embeddings = self.token_embeddings(token_ids)
        for layer in self.layers:
            embeddings = layer(embeddings)
        return self.lm_head(self.ln_final(embeddings))

    def init_cache(self, batch_size: int) -> ModelCache:
        """Create preallocated empty KV caches for all layers."""
        cache: ModelCache = []
        for layer in self.layers:
            attn = layer.attn
            K_buf = jnp.zeros((batch_size, attn.num_heads,
                               self.context_length, attn.d_k))
            V_buf = jnp.zeros((batch_size, attn.num_heads,
                               self.context_length, attn.d_k))
            cache_len = jnp.array(0, dtype=jnp.int32)
            cache.append((K_buf, V_buf, cache_len))
        return cache

    def forward_with_cache(
        self,
        token_ids: jnp.ndarray,
        token_positions: jnp.ndarray,
        cache: ModelCache,
    ) -> tuple[jnp.ndarray, ModelCache]:
        """Cache-aware forward for autoregressive generation.

        Args:
            token_ids: (batch, new_len) — new token(s) to process.
            token_positions: (batch, new_len) — absolute positions.
            cache: List of per-layer (K_buf, V_buf, cache_len) caches,
                   created via ``init_cache``.

        Returns:
            (logits, new_cache) where logits is (batch, new_len, vocab_size).
        """
        embeddings = self.token_embeddings(token_ids)
        new_cache: ModelCache = []

        for layer, layer_cache in zip(self.layers, cache):
            embeddings, updated = layer.forward_with_cache(
                embeddings, token_positions, layer_cache)
            new_cache.append(updated)

        logits = self.lm_head(self.ln_final(embeddings))
        return logits, new_cache
