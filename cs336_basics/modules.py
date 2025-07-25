import torch
import math
from torch.nn.init import trunc_normal_
from einops import einsum
import einx
from cs336_basics.nn_utils import scaled_dot_product_attention


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2.0 / (in_features + out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            trunc_normal_(torch.empty((out_features, in_features),
                                      device=device, dtype=dtype),
                          mean=0.0, std=std, a=-3*std, b=3*std))

    def forward(self, x):
        return einsum(x, self.weight, '... i, o i-> ... o')


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(
            trunc_normal_(torch.empty((num_embeddings, embedding_dim),
                                      device=device, dtype=dtype),
                          mean=0.0, std=1.0, a=-3.0, b=3.0))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)


class SiLU(torch.nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int = 0, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        if d_ff == 0:
            d_ff = 8 * d_model // 3
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.silu = SiLU(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Compute the position indices and dimension indices
        position = torch.arange(max_seq_len, device=device)
        dim = torch.arange(0, d_k, 2, device=device)  # (d_k // 2,)

        # Compute the frequency for each dimension
        angle = einsum(position.float(), 1 / theta **
                       (dim.float() / d_k), "s, d -> s d")

        # Precompute sin and cos
        self.register_buffer("sin", torch.sin(angle), persistent=False)
        self.register_buffer("cos", torch.cos(angle), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len) or (seq_len,)
        # Gather sin and cos for the given positions
        # Assume token_positions is (seq_len,) or broadcastable to x.shape[:-1]
        sin = self.sin[token_positions]  # (..., seq_len, d_k)
        cos = self.cos[token_positions]  # (..., seq_len, d_k)

        # Split x into even and odd features
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # Apply rotary transformation using einsum for elementwise ops
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        # Interleave even and odd features
        x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1).reshape(*x.shape)
        return x_rot


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 rope: bool = False, theta: float = 0.0, max_seq_len: int = 0,
                 device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        if rope:
            self.rope = RoPE(theta=theta, d_k=self.d_k,
                             max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None
        self.device = device

        # Single weight for QKV: (hd_k + hd_k + hd_v, d_model)
        self.weight_qkv = torch.nn.Parameter(
            torch.randn(num_heads*self.d_k*3, d_model, device=device)
        )

        # Output projection
        self.weight_o = torch.nn.Parameter(
            torch.randn(d_model, num_heads*self.d_k, device=device)
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(-2)
        Wx = einx.dot("... s d_model, (h d) d_model -> ... h s d",
                      x, self.weight_qkv, h=self.num_heads*3)
        Q, K, V = Wx.split(self.num_heads, dim=-3)
        if self.rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=self.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        # Create causal mask for self-attention
        mask = torch.tril(torch.ones(seq_len, seq_len,
                          device=self.device, dtype=torch.bool))

        x = scaled_dot_product_attention(Q, K, V, mask)
        return einx.dot("... h s d, d_model (h d) -> ... s d_model",
                        x, self.weight_o)


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 theta: float, max_seq_len: int,
                 device=None, dtype=None):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model=d_model,
                                           num_heads=num_heads,
                                           rope=True, theta=theta,
                                           max_seq_len=max_seq_len,
                                           device=device)
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff,
                          device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int,
                 num_layers: int, num_heads: int, d_ff: int, theta: float,
                 device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size,
                                          embedding_dim=d_model,
                                          device=device, dtype=dtype)
        self.layers = torch.nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                             theta=theta, max_seq_len=context_length,
                             device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.context_length = context_length
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.device = device

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len)
        embeddings = self.token_embeddings(token_ids)
        for layer in self.layers:
            embeddings = layer(embeddings)

        return self.lm_head(self.ln_final(embeddings))
