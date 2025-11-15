"""Tests for Conditional Flow Matcher classes."""

# Author: Kilian Fatras <kilian.fatras@mila.quebec>

import math

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax import random

from jaxcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
    pad_t_like_x,
)
from jaxcfm.optimal_transport import OTPlanSampler

TEST_SEED = 1994
TEST_BATCH_SIZE = 128
SIGMA_CONDITION = {
    "sb_cfm": lambda x: x <= 0,
}


def random_samples(key, shape, batch_size=TEST_BATCH_SIZE):
    """Generate random samples of different dimensions."""
    if isinstance(shape, int):
        shape = [shape]
    key1, key2 = random.split(key)
    return [random.normal(key1, (batch_size, *shape)), random.normal(key2, (batch_size, *shape))]


def compute_xt_ut(method, x0, x1, t_given, sigma, epsilon):
    if method == "vp_cfm":
        sigma_t = sigma
        mu_t = jnp.cos(math.pi / 2 * t_given) * x0 + jnp.sin(math.pi / 2 * t_given) * x1
        computed_xt = mu_t + sigma_t * epsilon
        computed_ut = (
            math.pi
            / 2
            * (jnp.cos(math.pi / 2 * t_given) * x1 - jnp.sin(math.pi / 2 * t_given) * x0)
        )
    elif method == "t_cfm":
        sigma_t = 1 - (1 - sigma) * t_given
        mu_t = t_given * x1
        computed_xt = mu_t + sigma_t * epsilon
        computed_ut = (x1 - (1 - sigma) * computed_xt) / sigma_t

    elif method == "sb_cfm":
        sigma_t = sigma * jnp.sqrt(t_given * (1 - t_given))
        mu_t = t_given * x1 + (1 - t_given) * x0
        computed_xt = mu_t + sigma_t * epsilon
        computed_ut = (
            (1 - 2 * t_given)
            / (2 * t_given * (1 - t_given) + 1e-8)
            * (computed_xt - (t_given * x1 + (1 - t_given) * x0))
            + x1
            - x0
        )
    elif method in ["exact_ot_cfm", "i_cfm"]:
        sigma_t = sigma
        mu_t = t_given * x1 + (1 - t_given) * x0
        computed_xt = mu_t + sigma_t * epsilon
        computed_ut = x1 - x0

    return computed_xt, computed_ut


def get_flow_matcher(method, sigma):
    if method == "vp_cfm":
        fm = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    elif method == "t_cfm":
        fm = TargetConditionalFlowMatcher(sigma=sigma)
    elif method == "sb_cfm":
        fm = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma, ot_method="sinkhorn")
    elif method == "exact_ot_cfm":
        fm = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif method == "i_cfm":
        fm = ConditionalFlowMatcher(sigma=sigma)
    return fm


def sample_plan(key, method, x0, x1, sigma):
    if method == "sb_cfm":
        key, subkey = random.split(key)
        x0, x1 = OTPlanSampler(method="sinkhorn", reg=2 * (sigma**2)).sample_plan(subkey, x0, x1)
    elif method == "exact_ot_cfm":
        key, subkey = random.split(key)
        x0, x1 = OTPlanSampler(method="exact").sample_plan(subkey, x0, x1)
    return key, x0, x1


@pytest.mark.parametrize("method", ["vp_cfm", "t_cfm", "sb_cfm", "exact_ot_cfm", "i_cfm"])
# Test both integer and floating sigma
@pytest.mark.parametrize("sigma", [0.0, 5e-4, 0.5, 1.5, 0, 1])
@pytest.mark.parametrize("shape", [[1], [2], [1, 2], [3, 4, 5]])
def test_fm(method, sigma, shape):
    batch_size = TEST_BATCH_SIZE
    key = random.PRNGKey(TEST_SEED)

    if method in SIGMA_CONDITION.keys() and SIGMA_CONDITION[method](sigma):
        with pytest.raises(ValueError):
            get_flow_matcher(method, sigma)
        return

    FM = get_flow_matcher(method, sigma)
    key, subkey = random.split(key)
    x0, x1 = random_samples(subkey, shape, batch_size=batch_size)
    
    key, subkey = random.split(key)
    t, xt, ut, eps = FM.sample_location_and_conditional_flow(subkey, x0, x1, return_noise=True)
    _ = FM.compute_lambda(t)

    # For OT methods, sample_location_and_conditional_flow internally modifies x0 and x1
    # via sample_plan, so we can't easily verify with manual computation.
    # For non-OT methods, we can verify the computation.
    if method not in ["sb_cfm", "exact_ot_cfm"]:
        # Use the same t and eps that were returned to compute expected values
        t_given = t.reshape(-1, *([1] * (x0.ndim - 1)))
        sigma_pad = pad_t_like_x(sigma, x0)
        computed_xt, computed_ut = compute_xt_ut(method, x0, x1, t_given, sigma_pad, eps)

        # Use np.allclose for floating point comparisons
        assert np.allclose(ut, computed_ut, rtol=1e-5, atol=1e-6)
        assert np.allclose(xt, computed_xt, rtol=1e-5, atol=1e-6)

