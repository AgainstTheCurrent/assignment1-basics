#!/usr/bin/env python3
"""Train a TransformerLM on the OpenWebText (OWT) dataset.

OWT is ~40x larger than TinyStories, so we increase training steps
and adjust warmup/checkpoint frequency accordingly.
"""

from cs336_basics.trainer import load_data, train_transformer

from config import (
    OWT_TRAIN_TOKENS_PATH, OWT_VALID_TOKENS_PATH,
    OWT_VOCAB_SIZE, CONTEXT_LENGTH, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, THETA,
    LR, LR_MIN, WEIGHT_DECAY, BETAS, EPS,
    GRADIENT_CLIP_NORM,
    OWT_MODEL_DIR,
)

# OWT-specific overrides
OWT_NUM_STEPS = 20000
OWT_BATCH_SIZE = 64
OWT_WARMUP_ITERS = 800
OWT_CHECKPOINT_FREQ = 2000
OWT_EVAL_FREQ = 200

if __name__ == "__main__":
    train_data = load_data(OWT_TRAIN_TOKENS_PATH)
    eval_data = load_data(OWT_VALID_TOKENS_PATH)
    train_transformer(
        train_data=train_data,
        eval_data=eval_data,
        vocab_size=OWT_VOCAB_SIZE,
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
        num_steps=OWT_NUM_STEPS,
        batch_size=OWT_BATCH_SIZE,
        lr_use_cosine_schedule=True,
        lr_min=LR_MIN,
        warmup_iters=OWT_WARMUP_ITERS,
        gradient_clip_norm=GRADIENT_CLIP_NORM,
        checkpoint_dir=OWT_MODEL_DIR,
        checkpoint_freq=OWT_CHECKPOINT_FREQ,
        eval_freq=OWT_EVAL_FREQ,
    )
