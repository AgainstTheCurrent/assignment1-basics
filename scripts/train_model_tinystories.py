#!/usr/bin/env python3
"""Train a TransformerLM on the TinyStories dataset."""

from cs336_basics.trainer import load_data, train_transformer

from config import (
    TS_TRAIN_TOKENS_PATH, TS_VALID_TOKENS_PATH,
    TS_VOCAB_SIZE, CONTEXT_LENGTH, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, THETA,
    LR, LR_MIN, WEIGHT_DECAY, BETAS, EPS,
    GRADIENT_CLIP_NORM,
    TS_MODEL_DIR,
)

# TinyStories-specific overrides
TS_NUM_STEPS = 5000
TS_BATCH_SIZE = 256
TS_WARMUP_ITERS = 200
TS_CHECKPOINT_FREQ = 1000
TS_EVAL_FREQ = 100

if __name__ == "__main__":
    train_data = load_data(TS_TRAIN_TOKENS_PATH)
    eval_data = load_data(TS_VALID_TOKENS_PATH)
    train_transformer(
        train_data=train_data,
        eval_data=eval_data,
        vocab_size=TS_VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        theta=THETA,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=BETAS,
        eps=EPS,
        num_steps=TS_NUM_STEPS,
        batch_size=TS_BATCH_SIZE,
        lr_use_cosine_schedule=True,
        lr_min=LR_MIN,
        warmup_iters=TS_WARMUP_ITERS,
        gradient_clip_norm=GRADIENT_CLIP_NORM,
        checkpoint_dir=TS_MODEL_DIR,
        checkpoint_freq=TS_CHECKPOINT_FREQ,
        eval_freq=TS_EVAL_FREQ,
    )
