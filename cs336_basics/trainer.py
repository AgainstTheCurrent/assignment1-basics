"""
Shared transformer training logic.

This module contains the model-agnostic training loop, loss function,
and training/eval step implementations.  Dataset-specific scripts
(e.g. ``train_model_tinystories.py``, ``train_model_owt.py``) call
``train_transformer`` with their own data paths and hyperparameters.
"""

import os
from pathlib import Path

import jax
import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
from flax import nnx

import optax

from cs336_basics.modules import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizers import adamw, get_lr_cosine_schedule
from cs336_basics.training_utils import (
    get_batch, save_checkpoint, load_checkpoint, save_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_data(data_path: str) -> np.ndarray:
    """Load training/validation data from numpy file."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    print(f"Loaded data from {data_path}: shape {data.shape}")
    return data


def cosine_schedule_scale(
    max_lr: float, min_lr: float,
    warmup_iters: int, cosine_cycle_iters: int,
):
    """Return a JAX-compatible schedule fn for optax.scale_by_schedule."""
    def schedule_fn(step):
        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        return lr / max_lr
    return schedule_fn


def loss_fn(model: nnx.Module,
            batch: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
    x, y = batch
    logits = model(x)
    return cross_entropy(logits, y), logits


@nnx.jit
def train_step(
    batch: tuple[jnp.ndarray, jnp.ndarray],
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    gradient_clip_norm: float = 0.0,
):
    """Single training step with optional gradient clipping."""
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch)
    grads = gradient_clipping(grads, gradient_clip_norm)
    optimizer.update(model, grads)
    metrics.update(loss=loss, perplexity=jnp.exp(loss),
                   logits=logits, labels=batch[1])


@nnx.jit
def eval_step(
    batch: tuple[jnp.ndarray, jnp.ndarray],
    model: nnx.Module,
    metrics: nnx.MultiMetric,
):
    """Single evaluation step (no gradients, no optimizer update)."""
    loss, logits = loss_fn(model, batch)
    metrics.update(loss=loss, perplexity=jnp.exp(loss),
                   logits=logits, labels=batch[1])


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_transformer(
    # Dataset
    train_data: np.ndarray,
    eval_data: np.ndarray,
    # Model hyperparameters
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
    num_steps: int,
    batch_size: int,
    lr_use_cosine_schedule: bool = False,
    lr_min: float = 0.0,
    warmup_iters: int = 0,
    gradient_clip_norm: float = 1.0,
    # Checkpointing
    checkpoint_dir: str = "./checkpoints",
    checkpoint_freq: int = 1000,
    # Evaluation
    eval_freq: int = 100,
):
    """Train a TransformerLM from scratch (or resume from checkpoint)."""
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(device_mesh, axis_names=("batch",))
    with mesh:
        model = TransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            theta=theta,
            rngs=nnx.Rngs(0))
        # Build optax transform, optionally chaining cosine LR schedule
        tx = adamw(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        if lr_use_cosine_schedule:
            schedule_fn = cosine_schedule_scale(
                max_lr=lr, min_lr=lr_min,
                warmup_iters=warmup_iters, cosine_cycle_iters=num_steps)
            tx = optax.chain(tx, optax.scale_by_schedule(schedule_fn))
        optimizer = nnx.Optimizer(model, tx=tx, wrt=nnx.Param)
        rngs = nnx.Rngs(0)
        metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average('loss'),
            perplexity=nnx.metrics.Average('perplexity')
        )
        eval_metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average('loss'),
            perplexity=nnx.metrics.Average('perplexity')
        )
        sharding = NamedSharding(mesh, P('batch', None))
        os.makedirs(checkpoint_dir, exist_ok=True)

        # --- Resume from latest checkpoint if available ---
        start_step = 0
        ckpt_dir = Path(checkpoint_dir)
        existing = sorted(
            ckpt_dir.glob("checkpoint_*.ckpt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if existing:
            latest_ckpt = existing[-1]
            print(f"Resuming from checkpoint: {latest_ckpt}")
            start_step = load_checkpoint(latest_ckpt, model, optimizer)
            print(f"Resumed at step {start_step}")
        else:
            print("No checkpoint found, training from scratch.")

        for step in range(start_step, num_steps):
            batch = get_batch(train_data, batch_size, context_length, rngs())
            batch = jax.device_put(batch, sharding)
            train_step(batch, model, optimizer, metrics, gradient_clip_norm)
            if (step + 1) % checkpoint_freq == 0:
                checkpoint_path = ckpt_dir / f"checkpoint_{step + 1}.ckpt"
                save_checkpoint(model, optimizer, step + 1, checkpoint_path)
            if (step + 1) % eval_freq == 0:
                metrics_dict = metrics.compute()
                eval_batch = get_batch(eval_data, batch_size, context_length, rngs())
                eval_batch = jax.device_put(eval_batch, sharding)
                eval_step(eval_batch, model, eval_metrics)
                eval_dict = eval_metrics.compute()
                print(f"Step {step + 1}: train loss={metrics_dict['loss']:.4f}, "
                      f"train ppl={metrics_dict['perplexity']:.4f}, "
                      f"eval loss={eval_dict['loss']:.4f}, "
                      f"eval ppl={eval_dict['perplexity']:.4f}")
                metrics.reset()
                eval_metrics.reset()

        # --- Save model weights only (for inference) ---
        final_model_path = ckpt_dir / "model_final.ckpt"
        save_model(model, final_model_path)
        print(f"Final model saved to {final_model_path}")
