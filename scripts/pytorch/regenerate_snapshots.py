"""
Regenerate test snapshots using built-in Flax NNX modules as reference.

Uses nnx.Linear, nnx.Embed, nnx.RMSNorm (built-in) for all leaf operations.
RoPE, SDPA, and their composition are implemented with direct JAX math.

Run from project root:
    python3 scripts/regenerate_snapshots.py
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import torch
from pathlib import Path
from einops import rearrange

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = PROJECT / "tests" / "_snapshots"
FIXTURES = PROJECT / "tests" / "fixtures"

# ---------------------------------------------------------------------------
# Fixtures (must match tests/conftest.py exactly)
# ---------------------------------------------------------------------------
n_layers = 3
vocab_size = 10_000
batch_size = 4
n_queries = 12
n_keys = 16
n_heads = 4
d_head = 16
d_model = n_heads * d_head   # 64
d_ff = 128
theta = 10000.0

torch.manual_seed(1); q = torch.randn(batch_size, n_queries, d_model)
torch.manual_seed(2); k = torch.randn(batch_size, n_keys, d_model)
torch.manual_seed(3); v = torch.randn(batch_size, n_keys, d_model)
torch.manual_seed(4); in_embeddings = torch.randn(batch_size, n_queries, d_model)
torch.manual_seed(5); mask = torch.randn(batch_size, n_queries, n_keys) > 0.5
torch.manual_seed(6); in_indices = torch.randint(0, 10_000, (batch_size, n_queries))
pos_ids = torch.arange(0, n_queries)

# Load model weights (same logic as conftest.py ts_state_dict fixture)
state_dict = torch.load(FIXTURES / "ts_tests" / "model.pt", map_location="cpu")
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_jnp(t: torch.Tensor) -> jnp.ndarray:
    return jnp.array(t.detach().cpu().numpy())


def save_snapshot(name: str, arr: jnp.ndarray):
    np.savez(SNAPSHOT_DIR / f"{name}.npz", array=np.array(arr))
    print(f"  [OK] {name}")


def make_linear(in_f: int, out_f: int, weight: jnp.ndarray) -> nnx.Linear:
    """Create an nnx.Linear and set its kernel from a (out, in) weight."""
    lin = nnx.Linear(in_f, out_f, use_bias=False, rngs=nnx.Rngs(0))
    lin.kernel = nnx.Param(weight.T)            # kernel is (in, out)
    return lin


def make_rmsnorm(dim: int, weight: jnp.ndarray, eps: float = 1e-5) -> nnx.RMSNorm:
    rms = nnx.RMSNorm(dim, epsilon=eps, use_fast_variance=False, rngs=nnx.Rngs(0))
    rms.scale = nnx.Param(weight)
    return rms


def make_embed(num: int, dim: int, weight: jnp.ndarray) -> nnx.Embed:
    emb = nnx.Embed(num, dim, rngs=nnx.Rngs(0))
    emb.embedding = nnx.Param(weight)
    return emb


# ---------------------------------------------------------------------------
# Reference primitives (pure JAX math — no custom modules)
# ---------------------------------------------------------------------------
def ref_softmax(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    x_max = jnp.max(x, axis=axis, keepdims=True)
    e_x = jnp.exp(x - x_max)
    return e_x / jnp.sum(e_x, axis=axis, keepdims=True)


def ref_sdpa(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = jnp.einsum("...nd,...md->...nm", Q, K) / jnp.sqrt(jnp.array(d_k, dtype=Q.dtype))
    if mask is not None:
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    return jnp.einsum("...nm,...md->...nd", ref_softmax(scores), V)


class RefRoPE:
    """Plain-JAX RoPE — no nnx.Module dependency."""
    def __init__(self, theta_val: float, d_k: int, max_seq_len: int):
        position = jnp.arange(max_seq_len).astype(jnp.float32)
        dim = jnp.arange(0, d_k, 2).astype(jnp.float32)
        angle = jnp.einsum("s,d->sd", position, 1.0 / theta_val ** (dim / d_k))
        self.sin = jnp.sin(angle)
        self.cos = jnp.cos(angle)

    def __call__(self, x, token_positions):
        sin = self.sin[token_positions]
        cos = self.cos[token_positions]
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        return jnp.stack(
            (x_even * cos - x_odd * sin, x_even * sin + x_odd * cos),
            axis=-1,
        ).reshape(x.shape)


def ref_mhsa(x, q_w, k_w, v_w, o_w, n_h, d_m, rope_obj=None, token_positions=None):
    """Reference MHSA using fused QKV math + built-in Linear for projections."""
    d_k = d_m // n_h
    seq_len = x.shape[-2]

    qkv_w = jnp.concatenate([q_w, k_w, v_w], axis=0)  # (3*d_m, d_m)
    W = qkv_w.reshape(n_h * 3, d_k, d_m)
    Wx = jnp.einsum("...sd,hkd->...hsk", x, W)
    Q, K, V = jnp.split(Wx, 3, axis=-3)

    if rope_obj is not None:
        if token_positions is None:
            token_positions = jnp.arange(seq_len)
        Q = rope_obj(Q, token_positions)
        K = rope_obj(K, token_positions)

    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    attn_out = ref_sdpa(Q, K, V, causal)

    W_o = o_w.reshape(d_m, n_h, d_k)
    return jnp.einsum("...hsd,ohd->...so", attn_out, W_o)


def ref_swiglu(x, w1, w2, w3):
    """SwiGLU using built-in nnx.Linear + jax.nn.silu."""
    l1 = make_linear(d_model, d_ff, w1)
    l2 = make_linear(d_ff, d_model, w2)
    l3 = make_linear(d_model, d_ff, w3)
    return l2(jax.nn.silu(l1(x)) * l3(x))


def ref_transformer_block(x, weights, n_h, d_m, d_f, theta_val, max_sl):
    """Reference transformer block using built-in RMSNorm + Linear + JAX MHSA."""
    rope = RefRoPE(theta_val, d_m // n_h, max_sl)

    ln1 = make_rmsnorm(d_m, to_jnp(weights["ln1.weight"]))
    q_w = to_jnp(weights["attn.q_proj.weight"])
    k_w = to_jnp(weights["attn.k_proj.weight"])
    v_w = to_jnp(weights["attn.v_proj.weight"])
    o_w = to_jnp(weights["attn.output_proj.weight"])

    h = ln1(x)
    h = ref_mhsa(h, q_w, k_w, v_w, o_w, n_h, d_m, rope_obj=rope)
    x = x + h

    ln2 = make_rmsnorm(d_m, to_jnp(weights["ln2.weight"]))
    w1 = to_jnp(weights["ffn.w1.weight"])
    w2 = to_jnp(weights["ffn.w2.weight"])
    w3 = to_jnp(weights["ffn.w3.weight"])

    h2 = ref_swiglu(ln2(x), w1, w2, w3)
    return x + h2


# ---------------------------------------------------------------------------
# Snapshot generators
# ---------------------------------------------------------------------------

def gen_linear():
    w = to_jnp(state_dict["layers.0.ffn.w1.weight"])   # (d_ff, d_model)
    x = to_jnp(in_embeddings)
    lin = make_linear(d_model, d_ff, w)
    save_snapshot("test_linear", lin(x))


def gen_embedding():
    w = to_jnp(state_dict["token_embeddings.weight"])
    ids = to_jnp(in_indices)
    emb = make_embed(vocab_size, d_model, w)
    save_snapshot("test_embedding", emb(ids))


def gen_rmsnorm():
    w = to_jnp(state_dict["layers.1.ln1.weight"])
    x = to_jnp(in_embeddings)
    rms = make_rmsnorm(d_model, w)
    save_snapshot("test_rmsnorm", rms(x))


def gen_rope():
    x = to_jnp(in_embeddings)
    positions = to_jnp(pos_ids)
    rope = RefRoPE(theta, d_model, n_queries)
    save_snapshot("test_rope", rope(x, positions))


def gen_swiglu():
    w1 = to_jnp(state_dict["layers.0.ffn.w1.weight"])
    w2 = to_jnp(state_dict["layers.0.ffn.w2.weight"])
    w3 = to_jnp(state_dict["layers.0.ffn.w3.weight"])
    x = to_jnp(in_embeddings)
    save_snapshot("test_swiglu", ref_swiglu(x, w1, w2, w3))


def gen_sdpa():
    Q, K, V, M = to_jnp(q), to_jnp(k), to_jnp(v), to_jnp(mask)
    save_snapshot("test_scaled_dot_product_attention", ref_sdpa(Q, K, V, M))


def gen_4d_sdpa():
    Q, K, V = (
        rearrange(to_jnp(x), "(batch head) seq d -> batch head seq d", head=2)
        for x in (q, k, v)
    )
    M = rearrange(to_jnp(mask), "(batch head) query key -> batch head query key", head=2)
    save_snapshot("test_4d_scaled_dot_product_attention", ref_sdpa(Q, K, V, M))


def gen_mhsa():
    x = to_jnp(in_embeddings)
    q_w = to_jnp(state_dict["layers.0.attn.q_proj.weight"])
    k_w = to_jnp(state_dict["layers.0.attn.k_proj.weight"])
    v_w = to_jnp(state_dict["layers.0.attn.v_proj.weight"])
    o_w = to_jnp(state_dict["layers.0.attn.output_proj.weight"])
    save_snapshot(
        "test_multihead_self_attention",
        ref_mhsa(x, q_w, k_w, v_w, o_w, n_heads, d_model),
    )


def gen_mhsa_rope():
    x = to_jnp(in_embeddings)
    positions = to_jnp(rearrange(pos_ids, "seq -> 1 seq"))
    q_w = to_jnp(state_dict["layers.0.attn.q_proj.weight"])
    k_w = to_jnp(state_dict["layers.0.attn.k_proj.weight"])
    v_w = to_jnp(state_dict["layers.0.attn.v_proj.weight"])
    o_w = to_jnp(state_dict["layers.0.attn.output_proj.weight"])
    rope = RefRoPE(theta, d_model // n_heads, n_keys)
    save_snapshot(
        "test_multihead_self_attention_with_rope",
        ref_mhsa(x, q_w, k_w, v_w, o_w, n_heads, d_model,
                 rope_obj=rope, token_positions=positions),
    )


def gen_transformer_block():
    block_weights = {
        k.replace("layers.0.", ""): v
        for k, v in state_dict.items()
        if "layers.0." in k
    }
    x = to_jnp(in_embeddings)
    save_snapshot(
        "test_transformer_block",
        ref_transformer_block(x, block_weights, n_heads, d_model, d_ff, theta, n_keys),
    )


def gen_transformer_lm():
    ids = to_jnp(in_indices)
    emb = make_embed(vocab_size, d_model, to_jnp(state_dict["token_embeddings.weight"]))
    x = emb(ids)

    for layer_idx in range(n_layers):
        pfx = f"layers.{layer_idx}."
        layer_w = {
            k.replace(pfx, ""): v
            for k, v in state_dict.items()
            if k.startswith(pfx)
        }
        x = ref_transformer_block(x, layer_w, n_heads, d_model, d_ff, theta, n_keys)

    ln_final = make_rmsnorm(d_model, to_jnp(state_dict["ln_final.weight"]))
    lm_head = make_linear(d_model, vocab_size, to_jnp(state_dict["lm_head.weight"]))

    out = lm_head(ln_final(x))
    save_snapshot("test_transformer_lm", out)
    save_snapshot("test_transformer_lm_truncated_input", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Regenerating snapshots using built-in Flax NNX modules …")
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    gen_linear()
    gen_embedding()
    gen_rmsnorm()
    gen_rope()
    gen_swiglu()
    gen_sdpa()
    gen_4d_sdpa()
    gen_mhsa()
    gen_mhsa_rope()
    gen_transformer_block()
    gen_transformer_lm()
    print("Done — all snapshots regenerated.")
