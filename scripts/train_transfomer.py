#!/usr/bin/env python3
"""
Training script for TransformerLM model with command line arguments.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add the parent directory to sys.path to import cs336_basics
sys.path.append(str(Path(__file__).parent.parent))

# Import after adding to path to avoid lint warning  # noqa: E402
from cs336_basics.modules import TransformerLM  # noqa: E402
from cs336_basics.optimizers import AdamW  # noqa: E402
from cs336_basics.nn_utils import (  # noqa: E402
    cross_entropy, gradient_clipping
)
from cs336_basics.training_utils import (  # noqa: E402
    get_batch, save_checkpoint, load_checkpoint
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a TransformerLM model")

    # Model parameters (matching TransformerLM constructor)
    parser.add_argument("--vocab_size", type=int, required=True,
                        help="Size of the vocabulary")
    parser.add_argument("--context_length", type=int, required=True,
                        help="Maximum sequence length the model can handle")
    parser.add_argument("--d_model", type=int, required=True,
                        help="Model dimension (embedding size)")
    parser.add_argument("--num_layers", type=int, required=True,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, required=True,
                        help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, required=True,
                        help="Dimension of feed-forward network")
    parser.add_argument("--theta", type=float, default=10000.0,
                        help="Theta parameter for RoPE (default: 10000.0)")

    # Optimizer parameters (matching AdamW constructor)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.999],
                        help="Beta parameters for Adam (default: 0.9 0.999)")
    parser.add_argument("--eps", type=float, default=1e-8,
                        help="Epsilon parameter for Adam (default: 1e-8)")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    parser.add_argument("--max_iters", type=int, default=1000,
                        help="Maximum number of training iterations "
                             "(default: 1000)")
    parser.add_argument("--eval_interval", type=int, default=100,
                        help="How often to evaluate the model (default: 100)")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="How often to log training progress "
                             "(default: 10)")
    parser.add_argument("--save_interval", type=int, default=500,
                        help="How often to save checkpoints (default: 500)")

    # Data parameters
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data (numpy array of "
                             "token IDs)")
    parser.add_argument("--val_data", type=str,
                        help="Path to validation data (numpy array of "
                             "token IDs)")

    # Tokenizer parameters (optional)
    parser.add_argument("--vocab_file", type=str,
                        help="Path to vocabulary file for tokenizer")
    parser.add_argument("--merges_file", type=str,
                        help="Path to merges file for tokenizer")

    # Model checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints "
                             "(default: ./checkpoints)")
    parser.add_argument("--resume_from", type=str,
                        help="Path to checkpoint to resume training from")

    # Training options
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping "
                             "(default: 1.0)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cpu, cuda, mps) "
                             "(default: auto)")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float16", "float32", "bfloat16"],
                        help="Data type for model parameters "
                             "(default: float32)")
    parser.add_argument("--compile", action="store_true",
                        help="Compile the model with torch.compile")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Get the appropriate device based on argument and availability."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif (hasattr(torch.backends, "mps") and
              torch.backends.mps.is_available()):
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_arg)


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_str]


def load_data(data_path: str) -> np.ndarray:
    """Load training/validation data from numpy file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    print(f"Loaded data from {data_path}: shape {data.shape}")
    return data


def estimate_loss(model, train_data, val_data, batch_size, context_length,
                  device, eval_iters=100):
    """Estimate loss on training and validation data."""
    model.eval()
    losses = {}

    for split, data in [("train", train_data), ("val", val_data)]:
        if data is None:
            continue

        split_losses = []
        for _ in range(eval_iters):
            x, y = get_batch(data, batch_size, context_length, device)
            with torch.no_grad():
                logits = model(x)
                # Reshape for cross entropy:
                # (batch * seq_len, vocab_size) and (batch * seq_len,)
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = y.view(-1)
                loss = cross_entropy(logits_flat, targets_flat)
                split_losses.append(loss.item())

        losses[split] = np.mean(split_losses)

    model.train()
    return losses


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get device and dtype
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    # Load data
    train_data = load_data(args.train_data)
    val_data = load_data(args.val_data) if args.val_data else None

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize model
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        device=device,
        dtype=dtype
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        eps=args.eps
    )

    # Optionally compile the model
    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)

    # Resume from checkpoint if specified
    start_iter = 0
    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    # Training loop
    print("Starting training...")
    model.train()

    for iter_num in range(start_iter, args.max_iters):
        # Get batch
        x, y = get_batch(train_data, args.batch_size, args.context_length,
                         device)

        # Forward pass
        logits = model(x)

        # Compute loss
        # Reshape for cross entropy:
        # (batch * seq_len, vocab_size) and (batch * seq_len,)
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = y.view(-1)
        loss = cross_entropy(logits_flat, targets_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if args.gradient_clip_norm > 0:
            gradient_clipping(model.parameters(), args.gradient_clip_norm)

        # Optimizer step
        optimizer.step()

        # Logging
        if iter_num % args.log_interval == 0:
            print(f"Iter {iter_num:6d} | Loss: {loss.item():.4f}")

        # Evaluation
        if iter_num % args.eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(model, train_data, val_data,
                                   args.batch_size, args.context_length, device)
            print(f"Iter {iter_num:6d} | Train loss: {losses.get('train', 'N/A'):.4f} | "
                  f"Val loss: {losses.get('val', 'N/A'):.4f}")

        # Save checkpoint
        if iter_num % args.save_interval == 0 and iter_num > 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_{iter_num}.pt")
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint_path = os.path.join(
        args.checkpoint_dir, "final_checkpoint.pt")
    save_checkpoint(model, optimizer, args.max_iters, final_checkpoint_path)
    print(f"Saved final checkpoint: {final_checkpoint_path}")

    print("Training completed!")


if __name__ == "__main__":
    main()
