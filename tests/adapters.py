"""
Test adapters that bridge PyTorch test tensors <-> JAX modules.

Module-related adapters convert torch.Tensor to jnp.ndarray, run through the
JAX (Flax NNX) implementation in cs336_basics.modules, and convert back.

Non-module utilities (softmax, cross_entropy, gradient_clipping, optimizers,
checkpointing, BPE) remain PyTorch-native since they are not part of the JAX
module rewrite.
"""
from __future__ import annotations

import os
from typing import IO, Any, BinaryIO
from collections.abc import Iterable

# Enable 64-bit precision in JAX so float64 torch tensors aren't truncated
import jax
jax.config.update("jax_enable_x64", True)
from jaxtyping import Float, Int

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from cs336_basics.bpe_training import train_bpe
from cs336_basics.bpe_tokenizer import BPETokenizer
import cs336_basics.modules as modules
from cs336_basics.nn_utils import softmax as jax_softmax, cross_entropy as jax_cross_entropy, scaled_dot_product_attention as jax_sdpa
from cs336_basics.pytorch.nn_utils import gradient_clipping
import cs336_basics.optimizers as optimizers
import cs336_basics.pytorch.training_utils as pytorch_training_utils
import cs336_basics.training_utils as training_utils


class _OptaxAdamWShim(torch.optim.Optimizer):
    """torch.optim.Optimizer interface backed by the JAX/optax AdamW.

    Bridges the PyTorch test harness (which calls .zero_grad / .backward / .step)
    with the optax GradientTransformation stored in cs336_basics.optimizers.adamw.
    """

    def __init__(self, params, lr=1e-3, weight_decay=0.01,
                 betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)
        # Build optax transform & per-parameter state
        self._tx = optimizers.adamw(lr=lr, weight_decay=weight_decay,
                                    betas=betas, eps=eps)
        # Initialise optax states lazily (first step)
        self._optax_states: dict[int, Any] = {}

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                pid = id(p)
                jp = jnp.array(p.data.detach().cpu().numpy())
                jg = jnp.array(p.grad.data.detach().cpu().numpy())
                if pid not in self._optax_states:
                    self._optax_states[pid] = self._tx.init(jp)
                updates, new_state = self._tx.update(jg, self._optax_states[pid], jp)
                self._optax_states[pid] = new_state
                new_p = jp + updates
                p.data.copy_(torch.from_numpy(np.array(new_p)))
        return loss


# ---------------------------------------------------------------------------
# Helpers: torch <-> jax conversion
# ---------------------------------------------------------------------------

def _to_jnp(t: Tensor) -> jnp.ndarray:
    """Convert a PyTorch tensor to a JAX array (via numpy, zero-copy when possible)."""
    return jnp.array(t.detach().cpu().numpy())


def _to_torch(a: jnp.ndarray, dtype=None) -> Tensor:
    """Convert a JAX array to a PyTorch tensor."""
    out = torch.from_numpy(np.array(a, copy=True))
    if dtype is not None:
        out = out.to(dtype)
    return out


def _dummy_rngs() -> nnx.Rngs:
    """Return a throwaway Rngs (weights will be overwritten anyway)."""
    return nnx.Rngs(0)


def _cat_qkv_jnp(q, k, v, num_heads: int):
    """Concatenate Q/K/V and reshape to (3 * h, d_k, d_model)."""
    qkv = jnp.concatenate([q, k, v], axis=0)
    d_model = qkv.shape[-1]
    d_k = q.shape[0] // num_heads
    return qkv.reshape(num_heads * 3, d_k, d_model)


def _reshape_o_proj_jnp(o_proj, num_heads: int):
    """Reshape output projection to (d_model, h, d_k)."""
    d_model = o_proj.shape[0]
    d_k = o_proj.shape[1] // num_heads
    return o_proj.reshape(d_model, num_heads, d_k)


# ---------------------------------------------------------------------------
# Module adapters (torch -> JAX -> torch)
# ---------------------------------------------------------------------------

def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    linear = modules.Linear(in_features=d_in, out_features=d_out, rngs=_dummy_rngs())
    linear.weight = nnx.Param(_to_jnp(weights))
    return _to_torch(linear(_to_jnp(in_features)), dtype=in_features.dtype)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    embedding = modules.Embedding(num_embeddings=vocab_size, embedding_dim=d_model, rngs=_dummy_rngs())
    embedding.weight = nnx.Param(_to_jnp(weights))
    return _to_torch(embedding(_to_jnp(token_ids)), dtype=weights.dtype)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    swiglu = modules.SwiGLU(d_model=d_model, d_ff=d_ff, rngs=_dummy_rngs())
    swiglu.w1.weight = nnx.Param(_to_jnp(w1_weight))
    swiglu.w2.weight = nnx.Param(_to_jnp(w2_weight))
    swiglu.w3.weight = nnx.Param(_to_jnp(w3_weight))
    return _to_torch(swiglu(_to_jnp(in_features)), dtype=in_features.dtype)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    jQ, jK, jV = _to_jnp(Q), _to_jnp(K), _to_jnp(V)
    jmask = _to_jnp(mask) if mask is not None else None
    out = jax_sdpa(jQ, jK, jV, jmask)
    return _to_torch(out, dtype=Q.dtype)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    mhsa = modules.MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads, rngs=_dummy_rngs())
    qkv = _cat_qkv_jnp(_to_jnp(q_proj_weight), _to_jnp(k_proj_weight), _to_jnp(v_proj_weight), num_heads)
    mhsa.weight_qkv = nnx.Param(qkv)
    mhsa.weight_o = nnx.Param(_reshape_o_proj_jnp(_to_jnp(o_proj_weight), num_heads))
    return _to_torch(mhsa(_to_jnp(in_features)), dtype=in_features.dtype)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    mhsa = modules.MultiHeadSelfAttention(
        d_model=d_model, num_heads=num_heads,
        rope=True, theta=theta, max_seq_len=max_seq_len, rngs=_dummy_rngs())
    qkv = _cat_qkv_jnp(_to_jnp(q_proj_weight), _to_jnp(k_proj_weight), _to_jnp(v_proj_weight), num_heads)
    mhsa.weight_qkv = nnx.Param(qkv)
    mhsa.weight_o = nnx.Param(_reshape_o_proj_jnp(_to_jnp(o_proj_weight), num_heads))
    jpos = _to_jnp(token_positions) if token_positions is not None else None
    return _to_torch(mhsa(_to_jnp(in_features), jpos), dtype=in_features.dtype)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    rope = modules.RoPE(theta=theta, d_k=d_k, max_seq_len=max_seq_len)
    return _to_torch(rope(_to_jnp(in_query_or_key), _to_jnp(token_positions)),
                     dtype=in_query_or_key.dtype)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    block = modules.TransformerBlock(
        d_model=d_model, num_heads=num_heads, d_ff=d_ff,
        theta=theta, max_seq_len=max_seq_len, rngs=_dummy_rngs())

    qkv = _cat_qkv_jnp(
        _to_jnp(weights["attn.q_proj.weight"]),
        _to_jnp(weights["attn.k_proj.weight"]),
        _to_jnp(weights["attn.v_proj.weight"]),
        num_heads)
    block.attn.weight_qkv = nnx.Param(qkv)
    block.attn.weight_o = nnx.Param(_reshape_o_proj_jnp(_to_jnp(weights["attn.output_proj.weight"]), num_heads))
    block.ln1.weight = nnx.Param(_to_jnp(weights["ln1.weight"]))
    block.ffn.w1.weight = nnx.Param(_to_jnp(weights["ffn.w1.weight"]))
    block.ffn.w2.weight = nnx.Param(_to_jnp(weights["ffn.w2.weight"]))
    block.ffn.w3.weight = nnx.Param(_to_jnp(weights["ffn.w3.weight"]))
    block.ln2.weight = nnx.Param(_to_jnp(weights["ln2.weight"]))
    return _to_torch(block(_to_jnp(in_features)), dtype=in_features.dtype)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    model = modules.TransformerLM(
        vocab_size=vocab_size, context_length=context_length,
        d_model=d_model, num_layers=num_layers, num_heads=num_heads,
        d_ff=d_ff, theta=rope_theta, rngs=_dummy_rngs())

    model.token_embeddings.weight = nnx.Param(_to_jnp(weights["token_embeddings.weight"]))
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."
        block = model.layers[layer_idx]
        qkv = _cat_qkv_jnp(
            _to_jnp(weights[prefix + "attn.q_proj.weight"]),
            _to_jnp(weights[prefix + "attn.k_proj.weight"]),
            _to_jnp(weights[prefix + "attn.v_proj.weight"]),
            num_heads)
        block.attn.weight_qkv = nnx.Param(qkv)
        block.attn.weight_o = nnx.Param(_reshape_o_proj_jnp(_to_jnp(weights[prefix + "attn.output_proj.weight"]), num_heads))
        block.ln1.weight = nnx.Param(_to_jnp(weights[prefix + "ln1.weight"]))
        block.ffn.w1.weight = nnx.Param(_to_jnp(weights[prefix + "ffn.w1.weight"]))
        block.ffn.w2.weight = nnx.Param(_to_jnp(weights[prefix + "ffn.w2.weight"]))
        block.ffn.w3.weight = nnx.Param(_to_jnp(weights[prefix + "ffn.w3.weight"]))
        block.ln2.weight = nnx.Param(_to_jnp(weights[prefix + "ln2.weight"]))
    model.ln_final.weight = nnx.Param(_to_jnp(weights["ln_final.weight"]))
    model.lm_head.weight = nnx.Param(_to_jnp(weights["lm_head.weight"]))
    return _to_torch(model(_to_jnp(in_indices)))


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    rmsnorm = modules.RMSNorm(d_model=d_model, eps=eps, rngs=_dummy_rngs())
    rmsnorm.weight = nnx.Param(_to_jnp(weights))
    return _to_torch(rmsnorm(_to_jnp(in_features)), dtype=in_features.dtype)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    silu = modules.SiLU()
    return _to_torch(silu(_to_jnp(in_features)), dtype=in_features.dtype)


# ---------------------------------------------------------------------------
# Non-module adapters (remain PyTorch-native)
# ---------------------------------------------------------------------------

def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # Tests rely on torch.Tensor output and device= kwarg; delegate to PyTorch impl
    return pytorch_training_utils.get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    return _to_torch(jax_softmax(_to_jnp(in_features), axis=dim), dtype=in_features.dtype)


def run_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    return _to_torch(jax_cross_entropy(_to_jnp(inputs), _to_jnp(targets)))


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    return gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    return _OptaxAdamWShim


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    return optimizers.get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    # Tests use torch.nn.Module / torch.optim.Optimizer; delegate to PyTorch impl
    pytorch_training_utils.save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    # Tests use torch.nn.Module / torch.optim.Optimizer; delegate to PyTorch impl
    return pytorch_training_utils.load_checkpoint(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        **kwargs,
    )
    return vocab, merges
