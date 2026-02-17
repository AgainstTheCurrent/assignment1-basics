#!/usr/bin/env python3
"""Generate text with the TinyStories model."""

from config import (
    TS_MODEL_PATH, TS_VOCAB_PATH, TS_MERGES_PATH,
    TS_VOCAB_SIZE, CONTEXT_LENGTH, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, THETA,
)
from cs336_basics.inference import run_inference

if __name__ == "__main__":
    run_inference(
        model_path=TS_MODEL_PATH,
        vocab_path=TS_VOCAB_PATH,
        merges_path=TS_MERGES_PATH,
        prompt="Once upon a time",
        vocab_size=TS_VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        theta=THETA,
    )
