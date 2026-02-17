"""
JAX implementation of training utilities.

Original PyTorch version preserved at cs336_basics/pytorch/training_utils.py.
"""

import os
from pathlib import Path
from typing import IO, Any, BinaryIO

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp


def get_batch(
    dataset: np.ndarray, batch_size: int, context_length: int,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a random batch of (input, target) pairs from *dataset*.

    Args:
        dataset: 1-D numpy array of token ids.
        batch_size: Number of sequences per batch.
        context_length: Length of each sequence.
        key: JAX PRNG key for random sampling.

    Returns:
        (x, y) where x has shape (batch_size, context_length) and y is x
        shifted right by one position.
    """
    max_start = dataset.shape[0] - context_length
    # Use numpy RNG to avoid JAX int32 overflow on large datasets (>2B tokens).
    # Convert JAX key to a numpy seed deterministically.
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    idx_np = rng.integers(0, max_start, size=batch_size)
    x = np.stack([dataset[i : i + context_length] for i in idx_np])
    y = np.stack([dataset[i + 1 : i + context_length + 1] for i in idx_np])
    return jnp.asarray(x, dtype=jnp.int32), jnp.asarray(y, dtype=jnp.int32)


def save_checkpoint(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    iteration: int,
    out: os.PathLike,
):
    """Serialize model, optimizer state and iteration number.

    Args:
        model: Flax NNX module to serialize.
        optimizer: Flax NNX optimizer wrapping the model.
        iteration: Current training iteration count.
        out: Path for the checkpoint.
    """
    _, model_state = nnx.split(model)
    _, opt_state = nnx.split(optimizer)
    ckpt = {
        "model_state": model_state,
        "optimizer_state": opt_state,
        "iteration": iteration,
    }
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(Path(out).resolve(), ckpt, force=True)
    checkpointer.wait_until_finished()


def save_model(
    model: nnx.Module,
    out: os.PathLike,
):
    """Serialize only the model weights (no optimizer state).

    This produces a smaller file suitable for inference / sharing.

    Args:
        model: Flax NNX module to serialize.
        out: Path for the saved model.
    """
    _, model_state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(Path(out).resolve(), {"model_state": model_state}, force=True)
    checkpointer.wait_until_finished()


def load_model(
    src: os.PathLike,
    model: nnx.Module,
):
    """Restore model weights from a model-only checkpoint.

    Args:
        src: Path of the saved model (created by :func:`save_model`).
        model: Flax NNX module whose state will be restored.
    """
    _, abstract_model_state = nnx.split(model)
    target = {"model_state": abstract_model_state}
    checkpointer = ocp.StandardCheckpointer()
    ckpt = checkpointer.restore(Path(src).resolve(), target)
    nnx.update(model, ckpt["model_state"])


def load_checkpoint(
    src: os.PathLike,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
) -> int:
    """Restore a serialized checkpoint into *model* and *optimizer*.

    Args:
        src: Path of the serialized checkpoint.
        model: Flax NNX module whose state will be restored.
        optimizer: Flax NNX optimizer whose state will be restored.

    Returns:
        The previously-serialized iteration number.
    """
    graph_def_model, abstract_model_state = nnx.split(model)
    graph_def_opt, abstract_opt_state = nnx.split(optimizer)
    target = {
        "model_state": abstract_model_state,
        "optimizer_state": abstract_opt_state,
        "iteration": 0,
    }
    checkpointer = ocp.StandardCheckpointer()
    ckpt = checkpointer.restore(Path(src).resolve(), target)
    nnx.update(model, ckpt["model_state"])
    nnx.update(optimizer, ckpt["optimizer_state"])
    return ckpt["iteration"]
