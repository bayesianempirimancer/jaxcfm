import copy
import os

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from torchvision.utils import save_image

# Note: NeuralODE from torchdyn would need to be replaced with a JAX ODE solver
# JAX has diffrax or jax.experimental.ode for ODE solving


def generate_samples(model, params, key, savedir, step, net_="normal", integration_steps=100):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    params:
        model parameters (Flax format)
    key:
        JAX PRNG key
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    # In JAX, we need to use an ODE solver
    # This would typically use diffrax or jax.experimental.ode
    # For now, this is a placeholder structure
    
    key, subkey = random.split(key)
    x0 = random.normal(subkey, (64, 3, 32, 32))
    
    # ODE integration would go here
    # Using a simple Euler method as placeholder
    dt = 1.0 / integration_steps
    x = x0
    for i in range(integration_steps):
        t = i * dt
        # Apply model - this would need proper Flax model call
        # v = model.apply(params, t, x)
        # x = x + dt * v
        pass
    
    # Clip and normalize
    x = jnp.clip(x, -1, 1)
    x = x / 2 + 0.5
    x_np = np.array(x)
    save_image(x_np, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)


def ema(source_params, target_params, decay):
    """Update target parameters using exponential moving average.
    
    In JAX/Flax, parameters are nested dictionaries.
    """
    def update_fn(targ, src):
        if isinstance(targ, dict):
            return {k: update_fn(targ[k], src[k]) for k in targ.keys()}
        else:
            return decay * targ + (1 - decay) * src
    
    from jax import tree_util
    return tree_util.tree_map(update_fn, target_params, source_params)


def infiniteloop(dataloader):
    """Infinite loop over dataloader, converting to JAX arrays."""
    while True:
        for x, y in iter(dataloader):
            # Convert torch tensors to JAX arrays
            x_jax = jnp.array(x.numpy())
            yield x_jax

