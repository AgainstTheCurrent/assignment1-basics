import jax
import jax.numpy as jnp
import optax

def learn(lr=1.0, iterations=10):
    key = jax.random.PRNGKey(0)
    weights = 5 * jax.random.normal(key, (10, 10))
    opt = optax.sgd(lr)
    opt_state = opt.init(weights)

    def loss_fn(w):
        return (w ** 2).mean()

    for t in range(iterations):
        loss, grads = jax.value_and_grad(loss_fn)(weights)
        print(loss.item())
        updates, opt_state = opt.update(grads, opt_state)
        weights = optax.apply_updates(weights, updates)


for lr in [1e1, 50, 100, 1e3]:
    learn(lr)
    print("=================")
