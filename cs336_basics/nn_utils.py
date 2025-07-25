from typing import Iterable
import torch
from einops import einsum
from jaxtyping import Float, Int
from torch import Tensor


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


def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"],
                  targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    inputs_max = inputs.max(dim=-1, keepdim=True).values
    inputs = inputs - inputs_max  # For numerical stability
    return (inputs.exp().sum(dim=-1).log() - inputs.gather(dim=-1, index=targets.long().unsqueeze(-1)).squeeze(-1)).mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    total_norm = torch.sqrt(sum(p.grad.data.norm()**2 for p in parameters if p.grad is not None))
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad = clip_coef * p.grad
