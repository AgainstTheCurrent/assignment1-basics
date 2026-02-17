"""Minimal single-batch overfit test to verify model + optimizer correctness."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from cs336_basics.modules import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizers import adamw
from cs336_basics.training_utils import get_batch


def loss_fn(model, batch):
    x, y = batch
    logits = model(x)
    return cross_entropy(logits, y)


model = TransformerLM(
    vocab_size=10000, context_length=64, d_model=128,
    num_layers=2, num_heads=4, d_ff=336, theta=10000,
    rngs=nnx.Rngs(0))

# Test with constant LR, no schedule, no weight decay, no gradient clipping
tx = adamw(lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)
optimizer = nnx.Optimizer(model, tx=tx, wrt=nnx.Param)

# Small batch for speed
data = np.memmap("../data/TinyStoriesV2-GPT4-train-tokens.txt", dtype=np.uint16, mode='r')
batch = get_batch(data, batch_size=2, context_length=64, key=jax.random.PRNGKey(42))

grad_fn = nnx.value_and_grad(loss_fn)

print(f"{'Step':>5} {'Loss':>10}")
print("-" * 20)
for step in range(500):
    loss, grads = grad_fn(model, batch)
    optimizer.update(model, grads)
    if step % 10 == 0:
        print(f"{step:5d} {float(loss):10.4f}")

print(f"\nFinal loss: {float(loss):.6f}")
print(f"Expected for memorization: ~0.0")
print(f"Random baseline (ln 10000): {float(jnp.log(jnp.array(10000.0))):.4f}")
