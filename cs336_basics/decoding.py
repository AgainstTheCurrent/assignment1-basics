"""
Text generation and decoding utilities for language models.
"""
import torch
import torch.nn.functional as F
from typing import Optional


def top_p_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    p: float,
    temperature: float = 1.0,
    stop_token_ids: list[int] = None,
) -> torch.Tensor:
    """
    Generate text using nucleus (top-p) sampling.
    
    Args:
        model: The language model to use for generation
        input_ids: Input token IDs of shape seq_len
        max_new_tokens: Maximum number of new tokens to generate
        p: Cumulative probability threshold for nucleus sampling
        temperature: Temperature for sampling (higher = more random)
        stop_token_id: Optional token ID to stop generation (e.g., end-of-text token)
    
    Returns:
        Generated token IDs of shape seq_len + max_new_tokens
    """
    model.eval()
    
    # Start with the input sequence
    generated = input_ids.clone()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get logits for the current sequence
            logits = model(generated)  # (seq_len, vocab_size)
            
            # Take the logits for the last token in each sequence
            next_token_logits = logits[-1, :]  # (vocab_size,)
            
            # Apply temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Convert to probabilities and sort
            probs = F.softmax(next_token_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            
            # Calculate cumulative probabilities
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff_index = torch.searchsorted(cumulative_probs, p)
            
            # Zero out probabilities outside the nucleus
            filtered_probs = sorted_probs[: cutoff_index + 1]
            
            # Renormalize
            filtered_probs = filtered_probs / cumulative_probs[cutoff_index]
            
            # Sample from the filtered distribution
            sampled_indices = torch.multinomial(filtered_probs, num_samples=1)
            next_token = torch.gather(sorted_indices, -1, sampled_indices)

            # Append to the generated sequence
            generated = torch.cat([generated, next_token], dim=0)

            # Stop generation if stop token is reached
            if stop_token_ids and next_token.item() in stop_token_ids:
                break

    return generated
