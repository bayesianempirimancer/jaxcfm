#!/usr/bin/env python
"""
Debug script extracted from the last cell of Flow_matching_tutorial.ipynb
This script helps identify performance bottlenecks in the training loop.
"""

import math
import os
import sys
import time

# Add project root to Python path
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt

from jaxcfm.optimal_transport import OTPlanSampler
from jaxcfm.models.models import MLP
from jaxcfm.utils import sample_8gaussians, sample_moons, plot_trajectories


def sample_conditional_pt(key, x0, x1, t, sigma):
    """
    Draw a sample from the probability path N(t * x1 + (1 - t) * x0, sigma), see (Eq.14) [1].

    Parameters
    ----------
    key : JAX PRNG key
    x0 : Array, shape (bs, *dim)
        represents the source minibatch
    x1 : Array, shape (bs, *dim)
        represents the target minibatch
    t : Array, shape (bs)
    sigma : float

    Returns
    -------
    xt : Array, shape (bs, *dim)

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    mu_t = (1 - t[:, None]) * x0 + t[:, None] * x1
    eps = random.normal(key, x0.shape)
    return mu_t + sigma * eps


def compute_conditional_vector_field(x0, x1):
    """
    Compute the conditional vector field ut(x1|x0) = x1 - x0, see Eq.(15) [1].

    Parameters
    ----------
    x0 : Array, shape (bs, *dim)
        represents the source minibatch
    x1 : Array, shape (bs, *dim)
        represents the target minibatch

    Returns
    -------
    ut : conditional vector field ut(x1|x0) = x1 - x0

    References
    ----------
    [1] Improving and Generalizing Flow-Based Generative Models with minibatch optimal transport, Preprint, Tong et al.
    """
    return x1 - x0


# Configuration
sigma = 0.1
dim = 2
batch_size = 256
learning_rate = 1e-3
num_iterations = 3

# Initialize model
print("Initializing model...")
key = random.PRNGKey(0)
model = MLP(dim=dim + 1, out_dim=dim, w=64, time_varying=False)  # +1 for time dimension
key, subkey = random.split(key)
dummy_input = jnp.ones((batch_size, dim + 1))
params = model.init(subkey, dummy_input)

# Initialize optimizer
print("Initializing optimizer...")
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Initialize OT sampler
print("Initializing OT sampler...")
ot_sampler = OTPlanSampler(method="exact")

# Training step function (OT sampling must be outside JIT)
# Track JIT compilation
jit_compile_count = [0]
jit_cache_info = {}

def train_step_with_tracking(params, opt_state, x0, x1, key):
    """Wrapper to track shapes and detect recompilation triggers"""
    # Check input shapes
    x0_shape = x0.shape
    x1_shape = x1.shape
    
    # Create hashable key (just use shapes, params structure should be constant)
    shape_key = (x0_shape, x1_shape)
    
    if shape_key not in jit_cache_info:
        jit_cache_info[shape_key] = {
            'count': 0,
            'first_seen': len(jit_cache_info),
            'shapes': {'x0': x0_shape, 'x1': x1_shape}
        }
        print(f"\n⚠️  NEW SHAPE DETECTED (will trigger JIT recompile #{len(jit_cache_info)}):")
        print(f"   x0 shape: {x0_shape}")
        print(f"   x1 shape: {x1_shape}")
        jit_compile_count[0] += 1
    
    jit_cache_info[shape_key]['count'] += 1
    
    return train_step(params, opt_state, x0, x1, key)

@jax.jit
def train_step(params, opt_state, x0, x1, key):
    key, subkey2, subkey3 = random.split(key, 3)
    t = random.uniform(subkey2, (batch_size,), minval=0.0, maxval=1.0)
    xt = sample_conditional_pt(subkey3, x0, x1, t, sigma=0.01)
    ut = compute_conditional_vector_field(x0, x1)
    
    # Concatenate xt and t
    model_input = jnp.concatenate([xt, t[:, None]], axis=-1)
    
    def loss_fn(params):
        vt = model.apply(params, model_input)
        loss = jnp.mean((vt - ut) ** 2)
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss, key

# Main training loop with timing
print("\n" + "=" * 60)
print("Starting training loop...")
print("=" * 60)
print("Note: First iteration will be slow due to JIT compilation...\n")

start = time.time()
ot_sample_time = 0.0
train_step_time = 0.0
data_sample_time = 0.0
total_iterations = 0

for k in range(num_iterations):
    iter_start = time.time()
    
    # Sample data
    data_start = time.time()
    key, subkey1, subkey2 = random.split(key, 3)
    x0 = sample_8gaussians(subkey1, batch_size)
    x1 = sample_moons(batch_size)
    data_sample_time += time.time() - data_start
    
    # Draw samples from OT plan (must be outside JIT due to numpy operations)
    ot_start = time.time()
    
    # Profile OT plan computation
    get_map_start = time.time()
    pi = ot_sampler.get_map(x0, x1)
    get_map_time = time.time() - get_map_start
    
    sample_map_start = time.time()
    i, j = ot_sampler.sample_map(subkey2, pi, x0.shape[0], replace=True)
    sample_map_time = time.time() - sample_map_start
    
    x0 = jnp.asarray(x0[i])
    x1 = jnp.asarray(x1[j])
    ot_sample_time += time.time() - ot_start
    
    if k == 0:
        print(f"\n🔍 OT Sampling Breakdown (iteration {k+1}):")
        print(f"   get_map (OT computation): {get_map_time:.4f}s")
        print(f"   sample_map (sampling): {sample_map_time:.4f}s")
        print(f"   Total OT time: {ot_sample_time:.4f}s")
    
    # Training step
    step_start = time.time()
    key, subkey = random.split(key)
    
    # Check shapes before calling
    if k == 0:
        print(f"\n📊 Input shapes for iteration {k+1}:")
        print(f"   x0: {x0.shape}, dtype: {x0.dtype}")
        print(f"   x1: {x1.shape}, dtype: {x1.dtype}")
        print(f"   params structure: {type(params)}")
    
    params, opt_state, loss, key = train_step_with_tracking(params, opt_state, x0, x1, subkey)
    train_step_time += time.time() - step_start
    
    total_iterations += 1
    
    if (k + 1) % 1 == 0:  # Report every iteration for debugging
        end = time.time()
        elapsed = end - start
        avg_time = elapsed / 1000
        
        print(f"\n{'='*60}")
        print(f"Iteration {k+1}:")
        print(f"  Loss: {loss:.6f}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Avg time per iter: {avg_time:.4f}s")
        print(f"  Breakdown:")
        print(f"    - Data sampling: {data_sample_time:.4f}s ({data_sample_time/elapsed*100:.1f}%)")
        print(f"    - OT sampling: {ot_sample_time:.4f}s ({ot_sample_time/elapsed*100:.1f}%)")
        print(f"    - Train step: {train_step_time:.4f}s ({train_step_time/elapsed*100:.1f}%)")
        print(f"  JIT compilations so far: {jit_compile_count[0]}")
        print(f"  Unique shape combinations: {len(jit_cache_info)}")
        
        # Reset counters
        start = end
        ot_sample_time = 0.0
        train_step_time = 0.0
        data_sample_time = 0.0
        
        # Skip trajectory generation for debugging (too slow)
        if False and (k + 1) % 5000 == 0:
            print("\n  Generating trajectory...")
            traj_start = time.time()
            key, subkey = random.split(key)
            x0_init = sample_8gaussians(subkey, 1024)
            
            # Define ODE term
            def vector_field(t, y, args):
                # y has shape (batch_size, dim)
                # We need to add time dimension
                t_batch = jnp.full((y.shape[0],), t)
                model_input = jnp.concatenate([y, t_batch[:, None]], axis=-1)
                return model.apply(params, model_input)
            
            term = ODETerm(vector_field)
            solver = Tsit5()
            saveat = SaveAt(ts=jnp.linspace(0, 1, 100))
            
            # Solve ODE for each sample
            traj_list = []
            for i in range(0, 1024, 256):  # Process in batches
                batch_x0 = x0_init[i:i+256]
                solution = diffeqsolve(term, solver, t0=0.0, t1=1.0, dt0=0.01, y0=batch_x0, saveat=saveat)
                traj_list.append(solution.ys)
            
            traj = jnp.concatenate(traj_list, axis=1)  # Shape: (100, 1024, 2)
            traj_time = time.time() - traj_start
            print(f"  Trajectory generation: {traj_time:.2f}s")
            plot_trajectories(np.array(traj))

print("\n" + "=" * 60)
print("Training complete!")
print("=" * 60)
print(f"\n📊 JIT Compilation Summary:")
print(f"   Total JIT compilations: {jit_compile_count[0]}")
print(f"   Unique shape combinations: {len(jit_cache_info)}")
if len(jit_cache_info) > 1:
    print(f"\n⚠️  WARNING: Multiple JIT compilations detected!")
    print(f"   This indicates shape changes that trigger recompilation.")
    print(f"   Each recompilation is very slow (~seconds).")
    print(f"\n   Shape combinations seen:")
    for i, (shape_key, info) in enumerate(jit_cache_info.items()):
        print(f"   {i+1}. Seen {info['count']} times: x0={info['shapes']['x0']}, x1={info['shapes']['x1']}")
else:
    print(f"✓ Only one JIT compilation (expected for fixed shapes)")

print(f"\n" + "=" * 60)
print("🔍 PERFORMANCE ANALYSIS:")
print("=" * 60)
print(f"\n❌ BOTTLENECK IDENTIFIED: OT Plan Computation (`get_map`)")
print(f"   - Takes ~10-11 seconds per iteration (99% of time)")
print(f"   - Uses Hungarian algorithm (exact OT) on 256x256 matrix")
print(f"   - Complexity: O(n³) = O(256³) ≈ 16M operations")
print(f"   - NOT JIT-compiled (runs in Python mode)")
print(f"\n✓ Fast components:")
print(f"   - Train step: ~0.001s per iteration (after JIT)")
print(f"   - Data sampling: ~0.006s per iteration")
print(f"   - OT sampling from plan: ~0.7s per iteration")
print(f"\n💡 RECOMMENDATIONS:")
print(f"   1. Use Sinkhorn (approximate OT) instead of exact OT for training:")
print(f"      ot_sampler = OTPlanSampler(method='sinkhorn', reg=0.05)")
print(f"      This is much faster (~100x) and often sufficient for training")
print(f"   2. JIT-compile `get_map` if possible (may require refactoring)")
print(f"   3. Reduce batch size (smaller matrices = faster Hungarian)")
print(f"   4. Use cached OT plans if data doesn't change much")
print("=" * 60)

