import jax.numpy as jnp
from flax import linen as nn
from jax import grad


class MLP(nn.Module):
    dim: int
    out_dim: int = None
    w: int = 64
    time_varying: bool = False

    def setup(self):
        if self.out_dim is None:
            self.out_dim = self.dim

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.w)(x)
        x = nn.selu(x)
        x = nn.Dense(self.w)(x)
        x = nn.selu(x)
        x = nn.Dense(self.w)(x)
        x = nn.selu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class GradModel:
    """Gradient model wrapper for JAX. In JAX, gradients are computed differently."""

    def __init__(self, action):
        self.action = action

    def __call__(self, x):
        # In JAX, we compute gradients using jax.grad
        # This returns a function that computes the gradient
        grad_fn = grad(lambda x: jnp.sum(self.action(x)))
        grad_output = grad_fn(x)
        return grad_output[:, :-1]

