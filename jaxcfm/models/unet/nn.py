"""Various utilities for neural networks."""

import math

import jax
import jax.numpy as jnp
from flax import linen as nn


# SiLU activation
def silu(x):
    """SiLU activation function."""
    return x * jax.nn.sigmoid(x)


class GroupNorm32(nn.Module):
    """Group normalization with 32 groups, converting to float32 for computation.
    
    Handles PyTorch-style (N, C, H, W) format by transposing to Flax's (N, H, W, C) format.
    """

    channels: int
    num_groups: int = 32

    @nn.compact
    def __call__(self, x):
        # Convert to float32 for computation, then back to original dtype
        original_dtype = x.dtype
        x_float32 = x.astype(jnp.float32)
        
        # Handle different input formats
        if len(x_float32.shape) == 2:
            # 1D: (N, C) - no transposition needed
            x_norm = nn.GroupNorm(num_groups=self.num_groups)(x_float32)
        elif len(x_float32.shape) == 3:
            # 1D spatial: (N, C, L) -> (N, L, C) for Flax
            x_transposed = jnp.transpose(x_float32, (0, 2, 1))
            x_norm = nn.GroupNorm(num_groups=self.num_groups)(x_transposed)
            x_norm = jnp.transpose(x_norm, (0, 2, 1))
        elif len(x_float32.shape) == 4:
            # 2D: (N, C, H, W) -> (N, H, W, C) for Flax
            x_transposed = jnp.transpose(x_float32, (0, 2, 3, 1))
            x_norm = nn.GroupNorm(num_groups=self.num_groups)(x_transposed)
            x_norm = jnp.transpose(x_norm, (0, 3, 1, 2))
        elif len(x_float32.shape) == 5:
            # 3D: (N, C, D, H, W) -> (N, D, H, W, C) for Flax
            x_transposed = jnp.transpose(x_float32, (0, 2, 3, 4, 1))
            x_norm = nn.GroupNorm(num_groups=self.num_groups)(x_transposed)
            x_norm = jnp.transpose(x_norm, (0, 4, 1, 2, 3))
        else:
            # Fallback: assume channels last already
            x_norm = nn.GroupNorm(num_groups=self.num_groups)(x_float32)
        
        return x_norm.astype(original_dtype)


def conv_nd(dims, in_channels=None, out_channels=None, kernel_size=None, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module.
    
    Handles PyTorch-style (N, C, ...) format by transposing to Flax's (N, ..., C) format.
    
    Args:
        dims: number of dimensions (1, 2, or 3)
        in_channels: input channels (optional, for compatibility)
        out_channels: output channels (required)
        kernel_size: kernel size (required)
        *args: additional positional args (for compatibility)
        **kwargs: additional keyword args
    """
    # Handle both old-style (dims, in_ch, out_ch, kernel_size) and new-style calls
    if in_channels is None and len(args) >= 1:
        # Old style: conv_nd(1, channels, channels*3, 1)
        in_channels = args[0] if len(args) > 0 else None
        out_channels = args[1] if len(args) > 1 else kwargs.pop('out_channels', None)
        kernel_size = args[2] if len(args) > 2 else kwargs.pop('kernel_size', None)
    
    if out_channels is None:
        raise ValueError("out_channels must be specified")
    if kernel_size is None:
        kernel_size = 3 if dims == 1 else (3, 3) if dims == 2 else (3, 3, 3)
    
    # Normalize kernel_size to tuple format
    if dims == 1:
        if isinstance(kernel_size, (list, tuple)):
            kernel_size = kernel_size[0] if len(kernel_size) > 0 else 1
        kernel_size_tuple = (kernel_size,)
        # Transpose: (N, C, L) <-> (N, L, C)
        transpose_in = (0, 2, 1)
        transpose_out = (0, 2, 1)
    elif dims == 2:
        if isinstance(kernel_size, int):
            kernel_size_tuple = (kernel_size, kernel_size)
        else:
            kernel_size_tuple = tuple(kernel_size)
        # Transpose: (N, C, H, W) <-> (N, H, W, C)
        transpose_in = (0, 2, 3, 1)
        transpose_out = (0, 3, 1, 2)
    elif dims == 3:
        if isinstance(kernel_size, int):
            kernel_size_tuple = (kernel_size, kernel_size, kernel_size)
        else:
            kernel_size_tuple = tuple(kernel_size)
        # Transpose: (N, C, D, H, W) <-> (N, D, H, W, C)
        transpose_in = (0, 2, 3, 4, 1)
        transpose_out = (0, 4, 1, 2, 3)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")
    
    class ConvWrapper(nn.Module):
        """Wrapper that handles PyTorch-style (N, C, ...) to Flax (N, ..., C) conversion."""
        features: int
        kernel_size: tuple
        transpose_in: tuple
        transpose_out: tuple
        kwargs: dict
        
        @nn.compact
        def __call__(self, x):
            # Transpose to Flax format: (N, C, ...) -> (N, ..., C)
            x_transposed = jnp.transpose(x, self.transpose_in)
            # Apply convolution
            conv = nn.Conv(features=self.features, kernel_size=self.kernel_size, **self.kwargs)
            out = conv(x_transposed)
            # Transpose back to PyTorch format: (N, ..., C) -> (N, C, ...)
            out_transposed = jnp.transpose(out, self.transpose_out)
            return out_transposed
    
    return ConvWrapper(
        features=out_channels,
        kernel_size=kernel_size_tuple,
        transpose_in=transpose_in,
        transpose_out=transpose_out,
        kwargs=kwargs
    )


def linear(*args, **kwargs):
    """Create a linear module."""
    out_features = kwargs.pop('out_features', args[0]) if args else kwargs.pop('features', None)
    return nn.Dense(features=out_features, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D average pooling module."""
    kernel_size = kwargs.pop('kernel_size', 2)
    stride = kwargs.pop('stride', 2)
    if dims == 1:
        return lambda x: nn.avg_pool(x, window_shape=(kernel_size,), strides=(stride,))
    elif dims == 2:
        return lambda x: nn.avg_pool(x, window_shape=(kernel_size, kernel_size), strides=(stride, stride))
    elif dims == 3:
        return lambda x: nn.avg_pool(x, window_shape=(kernel_size, kernel_size, kernel_size), strides=(stride, stride, stride))
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """Update target parameters to be closer to those of source parameters using an exponential
    moving average.

    :param target_params: the target parameter dict (Flax format).
    :param source_params: the source parameter dict (Flax format).
    :param rate: the EMA rate (closer to 1 means slower).
    """
    # In JAX/Flax, parameters are stored in nested dictionaries
    # We need to recursively update them
    def update_fn(targ, src):
        if isinstance(targ, dict):
            return {k: update_fn(targ[k], src[k]) for k in targ.keys()}
        else:
            return rate * targ + (1 - rate) * src
    
    return jax.tree_map(update_fn, target_params, source_params)


def zero_module(module):
    """Zero out the parameters of a module and return it.
    
    Note: In Flax, this would need to be done on the parameters dict.
    This is a placeholder that returns the module unchanged.
    """
    # In Flax, we can't modify parameters in place like PyTorch
    # This would need to be handled differently, e.g., by creating a new parameter dict
    return module


def scale_module(module, scale):
    """Scale the parameters of a module and return it.
    
    Note: In Flax, this would need to be done on the parameters dict.
    This is a placeholder that returns the module unchanged.
    """
    # In Flax, we can't modify parameters in place like PyTorch
    # This would need to be handled differently, e.g., by creating a new parameter dict
    return module


def mean_flat(tensor):
    """Take the mean over all non-batch dimensions."""
    return jnp.mean(tensor, axis=tuple(range(1, len(tensor.shape))))


def normalization(channels):
    """Make a standard normalization layer.

    :param channels: number of input channels.
    :return: a normalization layer.
    """
    # GroupNorm32 uses 32 groups, but we need to ensure channels is divisible by num_groups
    # If channels < 32, use channels as num_groups (which is effectively LayerNorm)
    # Otherwise use 32 groups
    if channels < 32:
        num_groups = channels
    else:
        num_groups = 32
    return GroupNorm32(channels=channels, num_groups=num_groups)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Array of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Array of positional embeddings.
    """
    half = dim // 2
    freqs = jnp.exp(
        -math.log(max_period)
        * jnp.arange(0, half, dtype=jnp.float32)
        / half
    )
    args = timesteps[:, None].astype(jnp.float32) * freqs[None, :]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    """Evaluate a function without caching intermediate activations, allowing for reduced memory at
    the expense of extra compute in the backward pass.

    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not explicitly take as
        arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        # In JAX, we use jax.checkpoint (formerly jax.remat) for gradient checkpointing
        return jax.checkpoint(func)(*inputs)
    else:
        return func(*inputs)

