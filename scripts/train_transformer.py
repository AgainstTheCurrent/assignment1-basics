#!/usr/bin/env python3
"""
Training script for TransformerLM model with command line arguments.
"""

import os
import sys
from pathlib import Path
import time

import numpy as np
import torch
import wandb

# Add the parent directory to sys.path to import cs336_basics
sys.path.append(str(Path(__file__).parent.parent))

# Import after adding to path to avoid lint warning  # noqa: E402
from cs336_basics.decoding import top_p_decode
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.modules import TransformerLM  # noqa: E402
from cs336_basics.optimizers import AdamW, get_lr_cosine_schedule  # noqa: E402
from cs336_basics.nn_utils import (  # noqa: E402
    cross_entropy, gradient_clipping
)
from cs336_basics.training_utils import (  # noqa: E402
    get_batch, save_checkpoint, load_checkpoint
)


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


def find_latest_checkpoint(checkpoint_dir: str) -> str:
    """Find the checkpoint with the largest iteration number in the directory."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("checkpoint_") and filename.endswith(".pt"):
            try:
                # Extract iteration number from filename like "checkpoint_1000.pt"
                iter_str = filename.replace(
                    "checkpoint_", "").replace(".pt", "")
                iteration = int(iter_str)
                checkpoint_files.append(
                    (iteration, os.path.join(checkpoint_dir, filename)))
            except ValueError:
                continue

    if not checkpoint_files:
        return None

    # Sort by iteration number and return the largest
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    latest_iteration, latest_path = checkpoint_files[0]

    print(
        f"Found latest checkpoint: {latest_path} (iteration {latest_iteration})")
    return latest_path


def train_transformer(
    # Model parameters
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    theta: float,
    # Optimizer parameters
    lr: float,
    weight_decay: float,
    betas: tuple,
    eps: float,
    # Training parameters
    batch_size: int,
    max_iters: int,
    eval_interval: int,
    log_interval: int,
    save_interval: int,
    # Data
    train_data: np.ndarray,
    val_data: np.ndarray = None,
    # Training options
    lr_use_cosine_schedule: bool = False,
    lr_min: float = 0.0,
    gradient_clip_norm: float = 1.0,
    device: torch.device = None,
    dtype: torch.dtype = None,
    compile_model: bool = False,
    checkpoint_dir: str = "./checkpoints",
    resume_from: str = None,
    seed: int = 42,
    # Wandb config
    wandb_project: str = "cs336-transformer",
    wandb_run_name: str = None,
    wandb_config: dict = None,
    target_val_loss: float = 1.45,
    patience_iters: int = 3000
) -> tuple[float, bool]:
    """
    Train a TransformerLM model with given hyperparameters.

    Returns:
        Tuple of (final_validation_loss, converged_successfully)
        converged_successfully is False if optimizer diverged (loss > 10.0) or no improvement for patience_iters
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set defaults
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float32

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    print(f"Target validation loss: {target_val_loss}")
    print(f"Early stopping patience: {patience_iters} iterations")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Auto-resume logic
    if resume_from is None:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            resume_from = latest_checkpoint
            print(f"Auto-resuming from: {resume_from}")
        else:
            print("No existing checkpoints found, starting from scratch")
    else:
        print(f"Manually resuming from: {resume_from}")

    # Initialize wandb
    if wandb_config is None:
        wandb_config = {
            "vocab_size": vocab_size,
            "context_length": context_length,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "theta": theta,
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
            "batch_size": batch_size,
            "max_iters": max_iters,
            "gradient_clip_norm": gradient_clip_norm,
            "target_val_loss": target_val_loss,
            "patience_iters": patience_iters,
            "resumed_from": resume_from if resume_from else "scratch",
        }

    # Generate run name if not provided
    if wandb_run_name is None:
        wandb_run_name = f"lr{lr}_bs{batch_size}"

    run = wandb.init(
        project=wandb_project,
        config=wandb_config,
        name=wandb_run_name,
        reinit=True
    )

    try:
        # Initialize model
        print("Initializing model...")
        model = TransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            theta=theta,
            device=device,
            dtype=dtype
        )

        # Count parameters
        param_layer = 2 * d_model + d_model * d_model * 4 + 3 * d_model * d_ff
        param_total = vocab_size * d_model + num_layers * param_layer + d_model + d_model * vocab_size
        print (f"Estimate total parameters: {param_total:,}")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad)
        print(f"Actual total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Estimate memory usage
        param_memory = total_params * 4
        adamw_memory = 2 * param_memory
        gradient_memory = param_memory
        activation_memory = context_length * batch_size * (num_layers *(16 * d_model + 2 * num_heads * context_length) + d_model + vocab_size + 1) * 4
        estimated_memory = param_memory + adamw_memory + gradient_memory + activation_memory
        estimated_memory /= (1024**3)  # Convert to GB
        print(f"Estimated GPU memory usage: {estimated_memory:.2f} GB")

        # Initialize optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )

        # Optionally compile the model
        if compile_model:
            print("Compiling model...")
            model = torch.compile(model)

        wandb.watch(model, log="all", log_freq=log_interval)

        # Resume from checkpoint if specified
        start_iter = 0
        if resume_from:
            print(f"Loading checkpoint: {resume_from}")
            start_iter = load_checkpoint(resume_from, model, optimizer)
            print(f"Resumed from iteration {start_iter}")

            # Log resumption info
            run.log({
                "resumed": True,
                "start_iteration": start_iter,
            }, step=start_iter)

        # Training loop
        print("Starting training...")
        model.train()
        start_time = time.time()
        final_val_loss = float('inf')
        converged = True
        best_val_loss = float('inf')

        # Early stopping variables
        last_improvement_iter = 0
        no_improvement_count = 0

        for iter_num in range(start_iter, max_iters):
            # Get batch
            x, y = get_batch(train_data, batch_size, context_length, device)

            # Forward pass
            logits = model(x)
            # Compute loss
            loss = cross_entropy(logits, y)

            # Check for divergence
            if loss.item() > 10.0 or torch.isnan(loss) or torch.isinf(loss):
                print(
                    f"DIVERGENCE DETECTED at iter {iter_num}: loss = {loss.item()}")
                converged = False
                run.log(
                    {"diverged": True, "divergence_reason": "high_loss"}, step=iter_num)
                break

            run.log({
                "train/loss": loss.item(),
                "train/wall_time": time.time() - start_time,
                "train/no_improvement_count": no_improvement_count,
            }, step=iter_num)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if gradient_clip_norm > 0:
                gradient_clipping(model.parameters(), gradient_clip_norm)

            # Optimizer step
            optimizer.step()
            if lr_use_cosine_schedule:
                optimizer.param_groups["lr"] = get_lr_cosine_schedule(
                    iter_num, lr, lr_min, 0, max_iters)

            # Logging
            if iter_num % log_interval == 0:
                print(f"Iter {iter_num:6d} | Train Loss: {loss.item():.4f} | "
                      f"No improvement: {no_improvement_count}")

            # Evaluation - focus on validation loss
            if iter_num % eval_interval == 0 and iter_num > 0:
                losses = estimate_loss(model, train_data, val_data,
                                       batch_size, context_length, device)
                val_loss = losses.get('val', float('inf'))
                train_loss = losses.get('train', loss.item())

                print(f"Iter {iter_num:6d} | Train: {train_loss:.4f} | "
                      f"Val: {val_loss:.4f} | Target: {target_val_loss} | "
                      f"Best: {best_val_loss:.4f}")

                # Check for improvement
                improved = False
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    last_improvement_iter = iter_num
                    improved = True
                    print(f"New best validation loss: {best_val_loss:.4f}")

                # Calculate no improvement count (always calculate, don't reset)
                no_improvement_count = iter_num - last_improvement_iter

                run.log({
                    "val/loss": val_loss,
                    "train/eval_loss": train_loss,
                    "val/best_loss": best_val_loss,
                    "val/target_achieved": val_loss <= target_val_loss,
                    "val/improved": improved,
                    "val/no_improvement_iters": no_improvement_count,
                }, step=iter_num)

                # Check for early stopping due to no improvement
                if no_improvement_count >= patience_iters:
                    print(
                        f"EARLY STOPPING: No improvement for {patience_iters} iterations")
                    print(
                        f"Last improvement at iteration {last_improvement_iter}")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    converged = False
                    run.log({
                        "diverged": True,
                        "divergence_reason": "no_improvement",
                        "last_improvement_iter": last_improvement_iter,
                        "no_improvement_count": no_improvement_count
                    }, step=iter_num)
                    final_val_loss = best_val_loss
                    break

                # Early stopping if target achieved
                if val_loss <= target_val_loss:
                    print(
                        f"Target validation loss {target_val_loss} achieved!")
                    final_val_loss = val_loss
                    break

                final_val_loss = val_loss

            # Save checkpoint
            if iter_num % save_interval == 0 and iter_num > 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_{iter_num}.pt")
                save_checkpoint(model, optimizer, iter_num, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

        # Final evaluation if not done recently
        if (max_iters - 1) % eval_interval != 0 and converged:
            losses = estimate_loss(model, train_data, val_data,
                                   batch_size, context_length, device)
            final_val_loss = losses.get('val', final_val_loss)

        # Save final checkpoint only if converged
        if converged:
            final_checkpoint_path = os.path.join(
                checkpoint_dir, "final_checkpoint.pt")
            save_checkpoint(model, optimizer, max_iters, final_checkpoint_path)
            print(f"Saved final checkpoint: {final_checkpoint_path}")

        # Final status
        if converged:
            print(
                f"Training completed successfully! Final validation loss: {final_val_loss:.4f}")
        else:
            print(
                f"Training stopped early. Best validation loss: {best_val_loss:.4f}")

        print(f"Converged: {converged}")

        return final_val_loss, converged

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"GPU OUT OF MEMORY with batch_size={batch_size}")
            torch.cuda.empty_cache()
            return float('inf'), False
        else:
            raise e
    finally:
        run.finish()


def estimate_gpu_memory_limit(device: torch.device) -> float:
    """Estimate available GPU memory in GB."""
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory
        return total_memory / (1024**3)
    return 24.0  # Assume 24GB if not CUDA


def find_max_batch_size(vocab_size: int, context_length: int, d_model: int,
                        num_layers: int, num_heads: int, d_ff: int,
                        device: torch.device, dtype: torch.dtype,
                        theta: float = 10000.0,
                        max_memory_gb: float = 22.0) -> int:
    """Find the maximum batch size that fits in GPU memory."""
    print(f"Finding max batch size for {max_memory_gb}GB memory limit...")

    # Start with a reasonable batch size and work up
    batch_size = 16
    max_batch_size = 16

    while batch_size <= 512:  # Reasonable upper limit
        try:
            # Create a temporary model to test memory usage
            torch.cuda.empty_cache()
            model = TransformerLM(
                vocab_size=vocab_size,
                context_length=context_length,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=theta,
                device=device,
                dtype=dtype
            )

            # Test with a forward pass
            x = torch.randint(0, vocab_size, (batch_size, context_length),
                              device=device)

            with torch.no_grad():
                _ = model(x)

            # Check memory usage
            if device.type == 'cuda':
                memory_used = torch.cuda.max_memory_allocated(
                    device) / (1024**3)
                if memory_used > max_memory_gb:
                    break

            max_batch_size = batch_size
            print(f"Batch size {batch_size}: OK")
            batch_size *= 2

            del model, x
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size}: OUT OF MEMORY")
                torch.cuda.empty_cache()
                break
            else:
                raise e

    print(f"Maximum batch size found: {max_batch_size}")
    return max_batch_size


def train_tinystories(
    # Override parameters (optional)
    vocab_size: int = 10000,
    context_length: int = 256,
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 16,
    d_ff: int = 1344,
    theta: float = 10000.0,
    lr: float = 0.0001,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    batch_size: int = 16,
    max_iters: int = 50000,
    eval_interval: int = 1000,
    log_interval: int = 1000,
    save_interval: int = 1000,
    train_data_path: str = "data/TinyStoriesV2-GPT4-train-tokens.txt",
    val_data_path: str = "data/TinyStoriesV2-GPT4-valid-tokens.txt",
    checkpoint_dir: str = "models/",
    device_str: str = "cuda",
    dtype_str: str = "float32",
    compile_model: bool = False,
    resume_from: str = None,
    seed: int = 42,
    target_val_loss: float = 1.45,
    wandb_run_name: str = None
) -> tuple[float, bool]:
    """Train on TinyStories dataset with hardcoded configuration."""

    # Initialize wandb
    wandb.login()

    # Get device and dtype
    device = get_device(device_str)
    dtype = get_dtype(dtype_str)

    # Load data
    train_data = load_data(train_data_path)
    val_data = load_data(val_data_path)

    if val_data is None:
        raise ValueError("Validation data required")

    return train_transformer(
        # Model parameters
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=theta,
        # Optimizer parameters
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        # Training parameters
        batch_size=batch_size,
        max_iters=max_iters,
        eval_interval=eval_interval,
        log_interval=log_interval,
        save_interval=save_interval,
        # Data
        train_data=train_data,
        val_data=val_data,
        # Training options
        gradient_clip_norm=1.0,
        device=device,
        dtype=dtype,
        compile_model=compile_model,
        checkpoint_dir=checkpoint_dir,
        resume_from=resume_from,
        seed=seed,
        wandb_project="cs336-assignment1",
        target_val_loss=target_val_loss,
        wandb_run_name=wandb_run_name
    )


def find_hyperparameters():
    """Hyperparameter sweep by calling train_tinystories with different parameters."""

    # Get device and dtype for finding max batch size
    device = get_device("cuda")
    dtype = get_dtype("float32")

    # Find maximum batch size for GPU memory
    max_batch_size = find_max_batch_size(
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        theta=10000.0,
        device=device,
        dtype=dtype,
        max_memory_gb=22.0  # Leave 2GB buffer on 24GB GPU
    )

    # Focused learning rate search around 0.003 with larger values
    learning_rates = [0.002, 0.003, 0.005, 0.006]

    # Generate batch sizes from 1 to memory limit, including typical sizes
    batch_sizes = [64, 128, 256, 512]
    batch_sizes = [bs for bs in batch_sizes if bs <= max_batch_size]

    # Add the maximum batch size if not already included
    if max_batch_size not in batch_sizes:
        batch_sizes.append(max_batch_size)

    batch_sizes.sort()

    print(f"Independent hyperparameter search:")
    print(
        f"Learning rates (focused around 3e-3 + larger values): {learning_rates}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Target validation loss: 1.45")

    best_val_loss = float('inf')
    best_params = {"lr": None, "batch_size": None}
    successful_runs = []

    # First sweep: Learning rates (with default batch size)
    print(f"\n{'='*80}")
    print("LEARNING RATE SWEEP (batch_size=16) - FOCUSED AROUND 3e-3 + LARGER VALUES")
    print(f"{'='*80}")

    best_lr = None
    for i, lr in enumerate(learning_rates):
        print(f"\n{'='*60}")
        print(f"LR RUN {i+1}/{len(learning_rates)}: lr={lr}")
        print(f"{'='*60}")

        try:
            name = f"bs_16_lr_{lr}"
            val_loss, converged = train_tinystories(
                lr=lr,
                batch_size=16,  # Use default batch size
                max_iters=10000,
                checkpoint_dir=f"models/{name}",
                seed=42 + i,
                wandb_run_name=name
            )

            result = {
                "type": "lr_sweep",
                "lr": lr,
                "batch_size": 16,
                "val_loss": val_loss,
                "converged": converged,
                "target_achieved": val_loss <= 1.45
            }

            successful_runs.append(result)

            print(f"Result: val_loss={val_loss:.4f}, converged={converged}, "
                  f"target_achieved={val_loss <= 1.45}")

            if converged and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lr = lr
                best_params["lr"] = lr
                print(f"NEW BEST LR: {lr} with val_loss={val_loss:.4f}")

        except Exception as e:
            print(f"FAILED: {e}")
            successful_runs.append({
                "type": "lr_sweep",
                "lr": lr,
                "batch_size": 16,
                "val_loss": float('inf'),
                "converged": False,
                "target_achieved": False,
                "error": str(e)
            })

    # Second sweep: Batch sizes (with best learning rate or default)
    print(f"\n{'='*80}")
    search_lr = best_lr if best_lr is not None else 3e-3  # Use best LR or 3e-3 default
    print(f"BATCH SIZE SWEEP (lr={search_lr})")
    print(f"{'='*80}")

    for i, batch_size in enumerate(batch_sizes):
        print(f"\n{'='*60}")
        print(f"BS RUN {i+1}/{len(batch_sizes)}: batch_size={batch_size}")
        print(f"{'='*60}")

        try:
            name = f"bs_{batch_size}_lr_{search_lr}"
            val_loss, converged = train_tinystories(
                lr=search_lr,
                batch_size=batch_size,
                max_iters=10000,
                checkpoint_dir=f"models/{name}",
                seed=42 + len(learning_rates) + i,
                wandb_run_name=name
            )

            result = {
                "type": "bs_sweep",
                "lr": search_lr,
                "batch_size": batch_size,
                "val_loss": val_loss,
                "converged": converged,
                "target_achieved": val_loss <= 1.45
            }

            successful_runs.append(result)

            print(f"Result: val_loss={val_loss:.4f}, converged={converged}, "
                  f"target_achieved={val_loss <= 1.45}")

            if converged and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params["batch_size"] = batch_size
                print(
                    f"NEW BEST BS: {batch_size} with val_loss={val_loss:.4f}")

        except Exception as e:
            print(f"FAILED: {e}")
            successful_runs.append({
                "type": "bs_sweep",
                "lr": search_lr,
                "batch_size": batch_size,
                "val_loss": float('inf'),
                "converged": False,
                "target_achieved": False,
                "error": str(e)
            })

    # Summary
    print(f"\n{'='*80}")
    print("INDEPENDENT HYPERPARAMETER SEARCH SUMMARY")
    print(f"{'='*80}")

    lr_runs = [r for r in successful_runs if r['type'] == 'lr_sweep']
    bs_runs = [r for r in successful_runs if r['type'] == 'bs_sweep']

    total_runs = len(lr_runs) + len(bs_runs)
    converged_runs = [r for r in successful_runs if r['converged']]
    target_achieved_runs = [r for r in successful_runs if r['target_achieved']]

    print(f"Total runs: {total_runs} (LR: {len(lr_runs)}, BS: {len(bs_runs)})")
    print(f"Converged runs: {len(converged_runs)}")
    print(f"Target achieved (≤1.45): {len(target_achieved_runs)}")

    print(f"\nBest learning rate: {best_params['lr']}")
    print(f"Best batch size: {best_params['batch_size']}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Show LR sweep results grouped by ranges
    print(f"\nLearning Rate Sweep Results (sorted by performance):")
    lr_converged = [r for r in lr_runs if r['converged']]
    if lr_converged:
        print("  Converged runs:")
        for run in sorted(lr_converged, key=lambda x: x['val_loss']):
            achieved = "✓" if run['target_achieved'] else "✗"
            print(f"    lr={run['lr']:>7}: {run['val_loss']:.4f} {achieved}")

        # Show performance in different ranges
        around_3e3 = [r for r in lr_converged if 2e-3 <= r['lr'] <= 4e-3]
        larger_values = [r for r in lr_converged if r['lr'] > 4e-3]

        if around_3e3:
            print(f"\n  Sweet spot range (2e-3 to 4e-3):")
            for run in sorted(around_3e3, key=lambda x: x['val_loss']):
                achieved = "✓" if run['target_achieved'] else "✗"
                print(
                    f"    lr={run['lr']:>7}: {run['val_loss']:.4f} {achieved}")

        if larger_values:
            print(f"\n  Larger learning rates (>4e-3):")
            for run in sorted(larger_values, key=lambda x: x['val_loss']):
                achieved = "✓" if run['target_achieved'] else "✗"
                print(
                    f"    lr={run['lr']:>7}: {run['val_loss']:.4f} {achieved}")
    else:
        print("  No converged runs")

    # Show BS sweep results
    print(f"\nBatch Size Sweep Results:")
    bs_converged = [r for r in bs_runs if r['converged']]
    if bs_converged:
        for run in sorted(bs_converged, key=lambda x: x['val_loss']):
            achieved = "✓" if run['target_achieved'] else "✗"
            print(
                f"  bs={run['batch_size']:>3}: {run['val_loss']:.4f} {achieved}")
    else:
        print("  No converged runs")

    # Show all runs that achieved target
    if target_achieved_runs:
        print(f"\nAll runs achieving target (≤1.45):")
        for run in sorted(target_achieved_runs, key=lambda x: x['val_loss']):
            print(
                f"  {run['type']}: lr={run['lr']}, bs={run['batch_size']}: {run['val_loss']:.4f}")

    # Final recommendation
    if best_params['lr'] and best_params['batch_size']:
        print(f"\nRecommended hyperparameters:")
        print(f"  Learning rate: {best_params['lr']}")
        print(f"  Batch size: {best_params['batch_size']}")
        print(f"  Expected validation loss: ≤{best_val_loss:.4f}")


def train_tinystories_main():
    """Train with best hyperparameters and auto-resume."""
    batch_size = 128
    name = f"tinystories_lr_0.003_bs_{batch_size}"
    val_loss, converged = train_tinystories(
        lr=0.003,
        batch_size=batch_size,
        max_iters=100000,
        checkpoint_dir=f"models/{name}",
        wandb_run_name=name,
        compile_model=True
    )

    print(f"Final result: val_loss={val_loss:.4f}, converged={converged}")


def generate():
    """Generate text using the trained model from train_tinystories_main."""
    # Model configuration (same as train_tinystories_main)
    vocab_size = 10000
    context_length = 256
    d_model = 512
    num_layers = 4
    num_heads = 16
    d_ff = 1344
    theta = 10000.0

    # Paths
    checkpoint_dir = "models/tinystories_lr_0.003_bs_16"
    vocab_path = "data/TinyStoriesV2-GPT4-vocab.json"
    merge_path = "data/TinyStoriesV2-GPT4-merges.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    print(f"Using device: {device}")
    print(f"Loading model from: {checkpoint_dir}")

    # Find the final checkpoint
    final_checkpoint_path = None
    checkpoint_candidates = [
        os.path.join(checkpoint_dir, "final_checkpoint.pt"),
        find_latest_checkpoint(checkpoint_dir)
    ]

    for candidate in checkpoint_candidates:
        if candidate and os.path.exists(candidate):
            final_checkpoint_path = candidate
            break

    if not final_checkpoint_path:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    print(f"Loading checkpoint: {final_checkpoint_path}")

    # Initialize model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        theta=theta,
        device=device,
        dtype=dtype
    )
    torch.set_float32_matmul_precision('high')
    compiled_model = torch.compile(model)

    # Load checkpoint (create dummy optimizer for loading)
    from cs336_basics.optimizers import AdamW
    dummy_optimizer = AdamW(model.parameters(), lr=0.001)

    try:
        iteration = load_checkpoint(
            final_checkpoint_path, model, dummy_optimizer)
        print(f"Loaded checkpoint from iteration {iteration}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = BPETokenizer.from_files(vocab_path, merge_path, special_tokens=["<|endoftext|>"])

    # Set model to evaluation mode
    model.eval()

    # Generate text with different prompts
    prompts = [
        "Once upon a time",
        "The little girl",
        "In a magical forest",
        "There was a brave knight"
    ]

    max_tokens = 256
    endoftext_token = tokenizer.encode("<|endoftext|>")

    for i, prompt in enumerate(prompts):
        print(f"\n{'-'*60}")
        print(f"PROMPT {i+1}: '{prompt}'")
        print(f"{'-'*60}")

        prompt_tokens = tokenizer.encode(prompt)

        if len(prompt_tokens) >= context_length:
            print(
                f"Prompt too long ({len(prompt_tokens)} tokens), truncating...")
            prompt_tokens = prompt_tokens[-context_length//2:]

        # Convert to tensor
        input_ids = torch.tensor(
            prompt_tokens, device=device, dtype=torch.long)

        # Generate using top-p decoding
        with torch.no_grad():
            # Generate tokens
            generated_ids = top_p_decode(
                compiled_model, input_ids, max_tokens - len(prompt_tokens),
                temperature=0.7, p=0.9,
                stop_token_ids=endoftext_token)

            all_tokens = generated_ids.cpu().tolist()

            # Decode to text
            full_text = tokenizer.decode(all_tokens)
            print(full_text)




if __name__ == "__main__":
    # Uncomment one of these:
    torch.set_float32_matmul_precision('high')
    # find_hyperparameters()  # Hyperparameter search
    # train_tinystories()  # Single training run
    train_tinystories_main()  # Train with best params and auto-resume
    # generate()  # Generate text using trained model
