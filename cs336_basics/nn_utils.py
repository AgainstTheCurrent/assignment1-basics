import torch
from einops import einsum


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    e_x = torch.exp(x - x_max)
    sum_e_x = torch.sum(e_x, dim=dim, keepdim=True)
    return e_x / sum_e_x


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    d_k = Q.size(-1)
    scores = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output
