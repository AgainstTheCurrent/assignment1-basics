"""
Shared constants for transformer training and inference.

Dataset-specific sections provide paths and model dirs for TinyStories
and OpenWebText (OWT).  Model architecture and training hyperparameters
are shared.
"""

# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------
DATA_DIR = "../data"

# ---------------------------------------------------------------------------
# TinyStories dataset
# ---------------------------------------------------------------------------
TS_TRAIN_TEXT_PATH = f"{DATA_DIR}/TinyStoriesV2-GPT4-train.txt"
TS_TRAIN_TOKENS_PATH = f"{DATA_DIR}/TinyStoriesV2-GPT4-train-tokens.txt"
TS_VALID_TOKENS_PATH = f"{DATA_DIR}/TinyStoriesV2-GPT4-valid-tokens.txt"
TS_VOCAB_PATH = f"{DATA_DIR}/TinyStoriesV2-GPT4-vocab.json"
TS_MERGES_PATH = f"{DATA_DIR}/TinyStoriesV2-GPT4-merges.json"
TS_MODEL_DIR = "../models/tinystories"
TS_MODEL_PATH = f"{TS_MODEL_DIR}/model_final.ckpt"

# ---------------------------------------------------------------------------
# OpenWebText (OWT) dataset
# ---------------------------------------------------------------------------
OWT_TRAIN_TEXT_PATH = f"{DATA_DIR}/owt_train.txt"
OWT_TRAIN_TOKENS_PATH = f"{DATA_DIR}/owt_train-tokens.txt"
OWT_VALID_TOKENS_PATH = f"{DATA_DIR}/owt_valid-tokens.txt"
OWT_VOCAB_PATH = f"{DATA_DIR}/owt-vocab.json"
OWT_MERGES_PATH = f"{DATA_DIR}/owt-merges.json"
OWT_MODEL_DIR = "../models/owt"
OWT_MODEL_PATH = f"{OWT_MODEL_DIR}/model_final.ckpt"

# ---------------------------------------------------------------------------
# BPE tokenizer
# ---------------------------------------------------------------------------
TS_VOCAB_SIZE = 10000
OWT_VOCAB_SIZE = 32000
SPECIAL_TOKENS = ["<|endoftext|>"]

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
CONTEXT_LENGTH = 256
D_MODEL = 512
NUM_LAYERS = 4
NUM_HEADS = 16
D_FF = 1344
THETA = 10000.0

# ---------------------------------------------------------------------------
# Training hyperparameters (shared across datasets)
# ---------------------------------------------------------------------------
LR = 3e-3
LR_MIN = 3e-4
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.999)
EPS = 1e-8
GRADIENT_CLIP_NORM = 1.0

