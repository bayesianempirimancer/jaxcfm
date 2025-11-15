import math

import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import random
from sklearn.datasets import make_moons

# Implement some helper functions


def eight_normal_sample(key, n, dim, scale=1, var=1):
    key1, key2 = random.split(key)
    # Use math.sqrt for compile-time constants (more efficient than jnp.sqrt)
    sqrt2_inv = 1.0 / math.sqrt(2)
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (sqrt2_inv, sqrt2_inv),
        (sqrt2_inv, -sqrt2_inv),
        (-sqrt2_inv, sqrt2_inv),
        (-sqrt2_inv, -sqrt2_inv),
    ]
    centers = jnp.array(centers) * scale
    noise = random.multivariate_normal(
        key1, jnp.zeros(dim), math.sqrt(var) * jnp.eye(dim), shape=(n,)
    )
    multi = random.choice(key2, 8, shape=(n,), replace=True)
    data = centers[multi] + noise
    return data


def sample_moons(n):
    x0, _ = make_moons(n_samples=n, noise=0.2, random_state=None)
    return jnp.array(x0 * 3 - 1)


def sample_8gaussians(key, n):
    return eight_normal_sample(key, n, 2, scale=5, var=0.1).astype(jnp.float32)


class jax_wrapper:
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        self.model = model

    def __call__(self, t, x, *args, **kwargs):
        # For JAX, we'll need to handle this differently
        # This is a compatibility wrapper, may need adjustment based on usage
        t_expanded = jnp.repeat(t, x.shape[0])[:, None]
        x_with_t = jnp.concatenate([x, t_expanded], axis=1)
        return self.model(x_with_t)


def plot_trajectories(traj):
    """Plot trajectories of some selected samples."""
    n = 2000
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black")
    plt.scatter(traj[:, :n, 0], traj[:, :n, 1], s=0.2, alpha=0.2, c="olive")
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=4, alpha=1, c="blue")
    plt.legend(["Prior sample z(S)", "Flow", "z(0)"])
    plt.xticks([])
    plt.yticks([])
    plt.show()

