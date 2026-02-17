#!/usr/bin/env python3
"""Generate text with the OpenWebText (OWT) model."""

from config import (
    OWT_MODEL_PATH, OWT_VOCAB_PATH, OWT_MERGES_PATH,
    OWT_VOCAB_SIZE, CONTEXT_LENGTH, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, THETA,
)
from cs336_basics.inference import run_inference

if __name__ == "__main__":
    run_inference(
        model_path=OWT_MODEL_PATH,
        vocab_path=OWT_VOCAB_PATH,
        merges_path=OWT_MERGES_PATH,
        prompt="The meaning of life is",
        vocab_size=OWT_VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        theta=THETA,
    )
