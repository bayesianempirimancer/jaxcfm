"""Helpers to train with 16-bit precision.

Note: JAX handles mixed precision differently than PyTorch. This file provides
compatibility utilities but JAX's native mixed precision support should be preferred.
"""

import jax
import jax.numpy as jnp

from . import logger

INITIAL_LOG_LOSS_SCALE = 20.0


def convert_params_to_f16(params):
    """Convert parameters to float16.
    
    :param params: Flax parameter dict.
    :return: Parameters converted to float16.
    """
    def convert_fn(x):
        if isinstance(x, jnp.ndarray) and x.dtype == jnp.float32:
            return x.astype(jnp.float16)
        return x
    
    return jax.tree_map(convert_fn, params)


def convert_params_to_f32(params):
    """Convert parameters to float32.
    
    :param params: Flax parameter dict.
    :return: Parameters converted to float32.
    """
    def convert_fn(x):
        if isinstance(x, jnp.ndarray) and x.dtype == jnp.float16:
            return x.astype(jnp.float32)
        return x
    
    return jax.tree_map(convert_fn, params)


def make_master_params(params):
    """Copy model parameters into full-precision parameters.
    
    :param params: Flax parameter dict (may be float16).
    :return: Full-precision parameter dict.
    """
    return convert_params_to_f32(params)


def update_master_params_from_model(master_params, model_params, loss_scale):
    """Update master parameters from model gradients.
    
    In JAX, this is typically handled by the optimizer, but this provides
    a compatibility layer for mixed precision training.
    
    :param master_params: Full-precision parameters.
    :param model_params: Model parameters (may be float16).
    :param loss_scale: Loss scaling factor.
    :return: Updated master parameters.
    """
    # In JAX, gradients are typically handled differently
    # This is a placeholder for compatibility
    return master_params


def check_overflow(value):
    """Check if a value has overflowed (inf or nan)."""
    return (value == float("inf")) or (value == -float("inf")) or (value != value)


class MixedPrecisionTrainer:
    """Mixed precision trainer for JAX.
    
    Note: JAX has native mixed precision support via jax.experimental.mixed_precision.
    This class provides a compatibility layer similar to PyTorch's approach.
    """
    
    def __init__(
        self,
        *,
        model,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
    ):
        self.model = model
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.lg_loss_scale = initial_lg_loss_scale

    def compute_loss_scale(self):
        """Get the current loss scale."""
        return 2 ** self.lg_loss_scale

    def adjust_loss_scale(self, has_overflow):
        """Adjust loss scale based on overflow detection."""
        if has_overflow:
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return False
        else:
            self.lg_loss_scale += self.fp16_scale_growth
            return True

    def compute_norms(self, params, grads, grad_scale=1.0):
        """Compute gradient and parameter norms.
        
        :param params: Parameter dict.
        :param grads: Gradient dict.
        :param grad_scale: Gradient scaling factor.
        :return: (grad_norm, param_norm) tuple.
        """
        def param_norm_fn(p):
            return jnp.sum(jnp.square(p.astype(jnp.float32)))
        
        def grad_norm_fn(g):
            if g is not None:
                return jnp.sum(jnp.square(g.astype(jnp.float32)))
            return 0.0
        
        param_norm = jnp.sqrt(sum(jax.tree_leaves(jax.tree_map(param_norm_fn, params))))
        grad_norm = jnp.sqrt(sum(jax.tree_leaves(jax.tree_map(grad_norm_fn, grads)))) / grad_scale
        
        return float(grad_norm), float(param_norm)

