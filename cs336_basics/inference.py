#!/usr/bin/env python3
"""
Shared inference logic for a trained TransformerLM model.

Dataset-specific scripts (``inference_tinystories.py``,
``inference_owt.py``) call ``run_inference`` with their own paths.
"""

import sys
import time

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from cs336_basics.modules import TransformerLM
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.decoding import generate_text
from cs336_basics.training_utils import load_model


def run_inference(
    model_path: str,
    vocab_path: str,
    merges_path: str,
    prompt: str,
    *,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    theta: float,
    max_new_tokens: int = 1000,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: int = 42,
):
    """Load a model and stream generated text to stdout."""
    # --- Load tokenizer ---------------------------------------------------
    print("Loading tokenizer...")
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path)

    # --- Build model ------------------------------------------------------
    print("Building model...")
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=theta,
        rngs=nnx.Rngs(0),
    )

    # --- Load model -------------------------------------------------------
    print(f"Loading model from {model_path}...")
    load_model(model_path, model)
    print("Model loaded.")

    # --- Encode prompt ----------------------------------------------------
    prompt_ids = tokenizer.encode(prompt)
    eos_id = tokenizer.token_to_id.get(b"<|endoftext|>")
    print(f"Prompt ({len(prompt_ids)} tokens): {prompt!r}")
    if eos_id is not None:
        print(f"EOS token id: {eos_id}")

    # --- Generate (streaming) --------------------------------------------
    rng = jax.random.PRNGKey(seed)
    print(f"Generating up to {max_new_tokens} tokens (KV-cached, top-p)...")
    print(f"{'='*60}")

    sys.stdout.write(prompt)
    sys.stdout.flush()

    t0 = time.perf_counter()
    num_generated = 0

    for token_id in generate_text(
        model,
        jnp.array(prompt_ids, dtype=jnp.int32),
        max_new_tokens=max_new_tokens,
        p=top_p,
        temperature=temperature,
        stop_token_ids=[eos_id] if eos_id is not None else None,
        rng=rng,
    ):
        text = tokenizer.decode([token_id])
        sys.stdout.write(text)
        sys.stdout.flush()
        num_generated += 1

    elapsed = time.perf_counter() - t0

    tok_per_sec = num_generated / elapsed if elapsed > 0 else float("inf")
    print(f"\n{'='*60}")
    print(f"Generated {num_generated} tokens in {elapsed:.2f}s "
          f"({tok_per_sec:.1f} tok/s)")
