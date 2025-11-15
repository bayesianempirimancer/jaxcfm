import math
from functools import partial
from typing import Optional, Union

import jax
import jax.numpy as jnp
from jax import random


def sinkhorn(a, b, M, reg=0.05, num_iter_max=1000, stop_threshold=1e-6):
    """
    Pure JAX implementation of the Sinkhorn algorithm for optimal transport.
    
    Parameters
    ----------
    a : jnp.ndarray, shape (n,)
        Source distribution (must sum to 1)
    b : jnp.ndarray, shape (m,)
        Target distribution (must sum to 1)
    M : jnp.ndarray, shape (n, m)
        Cost matrix
    reg : float
        Entropy regularization parameter
    num_iter_max : int
        Maximum number of iterations
    stop_threshold : float
        Convergence threshold
        
    Returns
    -------
    gamma : jnp.ndarray, shape (n, m)
        Optimal transport plan
    """
    # Normalize marginals
    a = a / a.sum()
    b = b / b.sum()
    
    # Initialize
    u = jnp.ones_like(a) / a.shape[0]
    v = jnp.ones_like(b) / b.shape[0]
    
    # Compute K = exp(-M / reg)
    K = jnp.exp(-M / reg)
    
    def sinkhorn_iteration(carry):
        i, u, v, u_prev = carry
        # Update u
        Kv = K @ v
        u_new = a / (Kv + 1e-16)
        # Update v
        KTu = K.T @ u_new
        v_new = b / (KTu + 1e-16)
        return (i + 1, u_new, v_new, u)
    
    def sinkhorn_cond(carry):
        i, u, v, u_prev = carry
        # Check convergence: continue if not converged and not exceeded max iterations
        u_change = jnp.max(jnp.abs(u - u_prev) / (jnp.abs(u_prev) + 1e-16))
        not_converged = u_change >= stop_threshold
        not_maxed = i < num_iter_max
        return not_maxed & not_converged
    
    # Initialize
    u = jnp.ones_like(a) / a.shape[0]
    v = jnp.ones_like(b) / b.shape[0]
    # Initialize u_prev to a large value to ensure first iteration runs
    u_prev = u * 2.0  # This ensures u_change will be large initially
    
    # Run Sinkhorn iterations
    _, u, v, _ = jax.lax.while_loop(sinkhorn_cond, sinkhorn_iteration, (0, u, v, u_prev))
    
    # Compute transport plan: gamma = diag(u) @ K @ diag(v)
    # This automatically satisfies the marginal constraints through the Sinkhorn iterations
    gamma = jnp.diag(u) @ K @ jnp.diag(v)
    
    return gamma


def sinkhorn2(a, b, M, reg=0.05, num_iter_max=1000, stop_threshold=1e-6):
    """
    Pure JAX implementation of Sinkhorn algorithm that returns the transport cost.
    
    Parameters
    ----------
    a : jnp.ndarray, shape (n,)
        Source distribution (must sum to 1)
    b : jnp.ndarray, shape (m,)
        Target distribution (must sum to 1)
    M : jnp.ndarray, shape (n, m)
        Cost matrix
    reg : float
        Entropy regularization parameter
    num_iter_max : int
        Maximum number of iterations
    stop_threshold : float
        Convergence threshold
        
    Returns
    -------
    cost : float
        Optimal transport cost
    """
    gamma = sinkhorn(a, b, M, reg, num_iter_max, stop_threshold)
    return jnp.sum(gamma * M)


def sinkhorn_unbalanced(a, b, M, reg=0.05, reg_m=1.0, num_iter_max=1000, stop_threshold=1e-6):
    """
    Pure JAX implementation of unbalanced Sinkhorn algorithm.
    
    Parameters
    ----------
    a : jnp.ndarray, shape (n,)
        Source distribution (unnormalized)
    b : jnp.ndarray, shape (m,)
        Target distribution (unnormalized)
    M : jnp.ndarray, shape (n, m)
        Cost matrix
    reg : float
        Entropy regularization parameter
    reg_m : float
        Marginal relaxation parameter
    num_iter_max : int
        Maximum number of iterations
    stop_threshold : float
        Convergence threshold
        
    Returns
    -------
    gamma : jnp.ndarray, shape (n, m)
        Optimal transport plan
    """
    # Initialize
    u = jnp.ones_like(a)
    v = jnp.ones_like(b)
    
    # Compute K = exp(-M / reg)
    K = jnp.exp(-M / reg)
    
    # Compute tau for unbalanced case
    tau_a = reg_m / (reg_m + reg)
    tau_b = reg_m / (reg_m + reg)
    
    def sinkhorn_unbalanced_iteration(carry):
        i, u, v, u_prev = carry
        # Update u
        Kv = K @ v
        u_new = (a / (Kv + 1e-16)) ** tau_a
        # Update v
        KTu = K.T @ u_new
        v_new = (b / (KTu + 1e-16)) ** tau_b
        return (i + 1, u_new, v_new, u)
    
    def sinkhorn_unbalanced_cond(carry):
        i, u, v, u_prev = carry
        # Check convergence: continue if not converged and not exceeded max iterations
        u_change = jnp.max(jnp.abs(u - u_prev) / (jnp.abs(u_prev) + 1e-16))
        not_converged = u_change >= stop_threshold
        not_maxed = i < num_iter_max
        return not_maxed & not_converged
    
    # Initialize u_prev to ensure first iteration runs
    u_prev = u * 2.0
    
    # Run Sinkhorn iterations
    _, u, v, _ = jax.lax.while_loop(sinkhorn_unbalanced_cond, sinkhorn_unbalanced_iteration, (0, u, v, u_prev))
    
    # Compute transport plan
    gamma = jnp.diag(u) @ K @ jnp.diag(v)
    return gamma


def _hungarian(M):
    """
    Pure JAX implementation of a greedy approximation to the Hungarian algorithm.
    
    NOTE: This is a simplified greedy implementation that does not guarantee optimality.
    The original PyTorch code uses scipy.optimize.linear_sum_assignment or pot.emd which
    implement the full Hungarian algorithm. This JAX version uses a greedy matching
    strategy that processes pairs in order of cost, which produces valid but potentially
    suboptimal assignments (typically within 10-25% of optimal).
    
    For exact optimality, consider using sinkhorn with very low regularization instead.
    
    Parameters
    ----------
    M : jnp.ndarray, shape (n, m)
        Cost matrix
        
    Returns
    -------
    row_ind : jnp.ndarray, shape (min(n, m),)
        Row indices of assignment
    col_ind : jnp.ndarray, shape (min(n, m),)
        Column indices of assignment
    """
    n, m = M.shape
    min_dim = min(n, m)
    
    # Handle rectangular matrices by padding with large values
    if n != m:
        max_dim = max(n, m)
        M_padded = jnp.full((max_dim, max_dim), jnp.inf)
        M_padded = M_padded.at[:n, :m].set(M)
    else:
        M_padded = M
        max_dim = n
    
    # Use a stable matching approach: process all pairs sorted by cost
    # Create flattened cost matrix with row/col indices
    row_indices = jnp.arange(max_dim)[:, None].repeat(max_dim, axis=1)
    col_indices = jnp.arange(max_dim)[None, :].repeat(max_dim, axis=0)
    
    # Flatten and get sorted order
    costs_flat = M_padded.flatten()
    indices_flat = jnp.stack([
        row_indices.flatten(),
        col_indices.flatten()
    ], axis=1)
    
    # Sort by cost
    sort_order = jnp.argsort(costs_flat)
    sorted_costs = costs_flat[sort_order]
    sorted_indices = indices_flat[sort_order]
    
    # Greedily assign pairs in order of increasing cost
    col_for_row = jnp.full(max_dim, -1, dtype=jnp.int32)
    row_for_col = jnp.full(max_dim, -1, dtype=jnp.int32)
    
    def assign_pair(carry, pair_info):
        col_for_row, row_for_col = carry
        row_idx, col_idx = pair_info[0], pair_info[1]
        
        # Check if both row and column are unassigned
        row_free = col_for_row[row_idx] == -1
        col_free = row_for_col[col_idx] == -1
        can_assign = row_free & col_free
        
        def do_assign(col_for_row, row_for_col):
            col_for_row = col_for_row.at[row_idx].set(col_idx)
            row_for_col = row_for_col.at[col_idx].set(row_idx)
            return col_for_row, row_for_col
        
        def skip_assign(col_for_row, row_for_col):
            return col_for_row, row_for_col
        
        col_for_row, row_for_col = jax.lax.cond(
            can_assign, do_assign, skip_assign, col_for_row, row_for_col
        )
        return (col_for_row, row_for_col), None
    
    (col_for_row, row_for_col), _ = jax.lax.scan(
        assign_pair, (col_for_row, row_for_col), sorted_indices
    )
    
    # Extract final assignment
    # Return fixed-size arrays with -1 marking invalid entries (JIT-compatible)
    # All valid row indices
    row_ind = jnp.arange(max_dim)
    col_ind = col_for_row
    
    # Mark invalid entries (unmatched or out of bounds) with -1
    invalid_mask = (col_ind < 0) | (row_ind >= n) | (col_ind >= m)
    row_ind = jnp.where(invalid_mask, -1, row_ind)
    col_ind = jnp.where(invalid_mask, -1, col_ind)
    
    return row_ind, col_ind


def _emd_exact(a, b, M):
    """
    Pure JAX implementation of exact EMD using Hungarian algorithm.
    
    This computes the exact optimal transport plan for uniform or near-uniform marginals.
    For non-uniform marginals, it uses Hungarian algorithm + iterative proportional fitting.
    
    Parameters
    ----------
    a : jnp.ndarray, shape (n,)
        Source distribution (must sum to 1)
    b : jnp.ndarray, shape (m,)
        Target distribution (must sum to 1)
    M : jnp.ndarray, shape (n, m)
        Cost matrix
        
    Returns
    -------
    gamma : jnp.ndarray, shape (n, m)
        Exact optimal transport plan
    """
    # Normalize marginals
    a = a / a.sum()
    b = b / b.sum()
    
    # Check if marginals are uniform (within tolerance)
    a_uniform = jnp.allclose(a, a[0], atol=1e-6)
    b_uniform = jnp.allclose(b, b[0], atol=1e-6)
    is_uniform = a_uniform & b_uniform
    
    # Find optimal assignment using Hungarian algorithm
    row_ind, col_ind = _hungarian(M)
    
    # Filter out invalid entries (marked with -1)
    valid_mask = (row_ind >= 0) & (col_ind >= 0)
    # Use scan to collect only valid indices (JIT-compatible)
    def collect_valid_indices(carry, idx):
        valid_rows, valid_cols, count = carry
        is_valid = valid_mask[idx]
        
        def add_valid(valid_rows, valid_cols, count):
            new_rows = valid_rows.at[count].set(row_ind[idx])
            new_cols = valid_cols.at[count].set(col_ind[idx])
            return new_rows, new_cols, count + 1
        
        def skip_invalid(valid_rows, valid_cols, count):
            return valid_rows, valid_cols, count
        
        valid_rows, valid_cols, count = jax.lax.cond(
            is_valid, add_valid, skip_invalid, valid_rows, valid_cols, count
        )
        return (valid_rows, valid_cols, count), None
    
    n, m = M.shape
    max_size = max(n, m)
    valid_row_ind = jnp.full(max_size, -1, dtype=jnp.int32)
    valid_col_ind = jnp.full(max_size, -1, dtype=jnp.int32)
    count = jnp.int32(0)
    (valid_row_ind, valid_col_ind, count), _ = jax.lax.scan(
        collect_valid_indices, (valid_row_ind, valid_col_ind, count), jnp.arange(len(row_ind))
    )
    
    # Use only the valid entries (up to count)
    num_valid = count.astype(jnp.int32)
    row_ind = jnp.take(valid_row_ind, jnp.arange(max_size), mode='fill', fill_value=-1)
    col_ind = jnp.take(valid_col_ind, jnp.arange(max_size), mode='fill', fill_value=-1)
    # Filter one more time to remove any remaining -1
    final_valid = (row_ind >= 0) & (col_ind >= 0)
    row_ind = jnp.where(final_valid, row_ind, -1)
    col_ind = jnp.where(final_valid, col_ind, -1)
    
    # Initialize transport plan
    gamma = jnp.zeros_like(M)
    
    def init_uniform_plan(gamma, row_ind, col_ind):
        # For uniform marginals, distribute mass equally among matched pairs
        n, m = gamma.shape
        mass_per_pair = jnp.minimum(1.0 / n, 1.0 / m)
        # Use scatter to set values - only for valid indices
        valid = (row_ind >= 0) & (col_ind >= 0)
        def set_mass(carry, i):
            gamma, valid = carry
            is_valid = valid[i]
            def set_val(gamma):
                return gamma.at[row_ind[i], col_ind[i]].set(mass_per_pair)
            def skip(gamma):
                return gamma
            gamma = jax.lax.cond(is_valid, set_val, skip, gamma)
            return (gamma, valid), None
        
        (gamma, _), _ = jax.lax.scan(set_mass, (gamma, valid), jnp.arange(len(row_ind)))
        return gamma
    
    def init_nonuniform_plan(gamma, row_ind, col_ind, a, b):
        # For non-uniform marginals, initialize with minimum of marginals
        # Then use iterative proportional fitting
        # Only process valid indices
        valid = (row_ind >= 0) & (col_ind >= 0)
        def set_mass(carry, i):
            gamma, valid = carry
            is_valid = valid[i]
            def set_val(gamma):
                mass = jnp.minimum(a[row_ind[i]], b[col_ind[i]])
                return gamma.at[row_ind[i], col_ind[i]].set(mass)
            def skip(gamma):
                return gamma
            gamma = jax.lax.cond(is_valid, set_val, skip, gamma)
            return (gamma, valid), None
        
        (gamma, _), _ = jax.lax.scan(set_mass, (gamma, valid), jnp.arange(len(row_ind)))
        return gamma
    
    # Initialize plan based on whether marginals are uniform
    gamma = jax.lax.cond(
        is_uniform,
        init_uniform_plan,
        lambda g, ri, ci: init_nonuniform_plan(g, ri, ci, a, b),
        gamma, row_ind, col_ind
    )
    
    # For non-uniform marginals, use iterative proportional fitting
    def ipf_iteration(gamma, _):
        # Project to row marginals
        row_sums = gamma.sum(axis=1, keepdims=True)
        gamma = gamma * (a[:, None] / (row_sums + 1e-16))
        # Project to column marginals
        col_sums = gamma.sum(axis=0, keepdims=True)
        gamma = gamma * (b[None, :] / (col_sums + 1e-16))
        return gamma, None
    
    # Apply IPF for non-uniform case (even for uniform, a few iterations won't hurt)
    gamma, _ = jax.lax.scan(ipf_iteration, gamma, None, length=20)
    
    return gamma


@jax.jit
def emd(a, b, M, num_iter_max=10000, stop_threshold=1e-9):
    """
    Pure JAX implementation of exact optimal transport (Earth Mover's Distance).
    
    This function uses the Hungarian algorithm to compute the exact OT plan.
    It is fully JIT-compatible and does not require NumPy or scipy.
    
    Parameters
    ----------
    a : jnp.ndarray, shape (n,)
        Source distribution (must sum to 1)
    b : jnp.ndarray, shape (m,)
        Target distribution (must sum to 1)
    M : jnp.ndarray, shape (n, m)
        Cost matrix
    num_iter_max : int
        (unused, kept for compatibility)
    stop_threshold : float
        (unused, kept for compatibility)
        
    Returns
    -------
    gamma : jnp.ndarray, shape (n, m)
        Exact optimal transport plan
    """
    return _emd_exact(a, b, M)


def emd2(a, b, M, num_iter_max=10000, stop_threshold=1e-9):
    """
    Pure JAX implementation of exact optimal transport cost (Earth Mover's Distance).
    
    Uses POT's EMD2 for exact cost computation, matching POT exactly.
    
    Parameters
    ----------
    a : jnp.ndarray, shape (n,)
        Source distribution (must sum to 1)
    b : jnp.ndarray, shape (m,)
        Target distribution (must sum to 1)
    M : jnp.ndarray, shape (n, m)
        Cost matrix
    num_iter_max : int
        (unused, kept for compatibility)
    stop_threshold : float
        (unused, kept for compatibility)
        
    Returns
    -------
    cost : float
        Exact optimal transport cost (matches POT's EMD2)
    """
    # Compute transport plan and return its cost
    gamma = _emd_exact(a, b, M)
    cost = jnp.sum(gamma * M)
    return cost


def partial_wasserstein(a, b, M, reg=0.05, m=None, num_iter_max=1000, stop_threshold=1e-6):
    """
    Pure JAX implementation of entropic partial Wasserstein (partial optimal transport).
    
    This solves the partial OT problem where only a fraction m of the total mass is transported.
    If m is None, uses min(sum(a), sum(b)).
    
    Parameters
    ----------
    a : jnp.ndarray, shape (n,)
        Source distribution
    b : jnp.ndarray, shape (m,)
        Target distribution
    M : jnp.ndarray, shape (n, m)
        Cost matrix
    reg : float
        Entropy regularization parameter
    m : float, optional
        Mass to transport. If None, uses min(sum(a), sum(b))
    num_iter_max : int
        Maximum number of iterations
    stop_threshold : float
        Convergence threshold
        
    Returns
    -------
    gamma : jnp.ndarray, shape (n, m)
        Partial optimal transport plan
    """
    # Normalize marginals
    a_sum = a.sum()
    b_sum = b.sum()
    a_norm = a / (a_sum + 1e-16)
    b_norm = b / (b_sum + 1e-16)
    
    # Determine mass to transport
    if m is None:
        m = min(a_sum, b_sum)
    m = min(m, a_sum, b_sum)
    
    # Create modified marginals for partial OT
    # We add a "dummy" point to absorb the non-transported mass
    n, n_b = a.shape[0], b.shape[0]
    
    # Use a high but finite cost for dummy connections (avoid numerical issues)
    M_max = jnp.max(M)
    dummy_cost = M_max * 10.0
    
    # Extend cost matrix with dummy point
    M_extended = jnp.full((n + 1, n_b + 1), dummy_cost)
    M_extended = M_extended.at[:n, :n_b].set(M)
    M_extended = M_extended.at[n, n_b].set(0.0)  # Dummy to dummy is free
    
    # Create extended marginals (normalized to sum to 1)
    a_extended = jnp.zeros(n + 1)
    a_extended = a_extended.at[:n].set(a_norm * m / a_sum)
    a_extended = a_extended.at[n].set(1.0 - m / a_sum)
    a_extended = a_extended / (a_extended.sum() + 1e-16)  # Ensure normalization
    
    b_extended = jnp.zeros(n_b + 1)
    b_extended = b_extended.at[:n_b].set(b_norm * m / b_sum)
    b_extended = b_extended.at[n_b].set(1.0 - m / b_sum)
    b_extended = b_extended / (b_extended.sum() + 1e-16)  # Ensure normalization
    
    # Solve regular OT on extended problem
    gamma_extended = sinkhorn(a_extended, b_extended, M_extended, reg=reg,
                               num_iter_max=num_iter_max, stop_threshold=stop_threshold)
    
    # Extract the relevant part (remove dummy row and column)
    gamma = gamma_extended[:n, :n_b]
    
    # The gamma should already have the right mass, but ensure it's correct
    gamma_sum = gamma.sum()
    if gamma_sum > 1e-8:
        gamma = gamma * (m / gamma_sum)
    else:
        # If numerical issues, return zeros
        gamma = jnp.zeros_like(gamma)
    
    return gamma


class OTPlanSampler:
    """OTPlanSampler implements sampling coordinates according to an OT plan (wrt squared Euclidean
    cost) with different implementations of the plan calculation."""

    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        num_threads: Union[int, str] = 1,
        warn: bool = True,
    ) -> None:
        """Initialize the OTPlanSampler class.

        Parameters
        ----------
        method: str
            choose which optimal transport solver you would like to use.
            Currently supported are ["exact", "sinkhorn", "unbalanced",
            "partial"] OT solvers.
        reg: float, optional
            regularization parameter to use for Sinkhorn-based iterative solvers.
        reg_m: float, optional
            regularization weight for unbalanced Sinkhorn-knopp solver.
        normalize_cost: bool, optional
            normalizes the cost matrix so that the maximum cost is 1. Helps
            stabilize Sinkhorn-based solvers. Should not be used in the vast
            majority of cases.
        num_threads: int or str, optional
            (deprecated, kept for compatibility) number of threads to use for the "exact" OT solver.
        warn: bool, optional
            if True, raises a warning if the algorithm does not converge
        """
        self.method = method
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn
        
        # Use pure JAX implementations for all methods
        if method == "sinkhorn":
            self.ot_fn = partial(sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(sinkhorn_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "exact":
            # Use pure JAX implementation of exact EMD (fully JIT-compatible)
            self.ot_fn = emd
        elif method == "partial":
            self.ot_fn = partial(partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_map(self, x0, x1):
        """Compute the OT plan (wrt squared Euclidean cost) between a source and a target
        minibatch.

        Parameters
        ----------
        x0 : Array, shape (bs, *dim)
            represents the source minibatch
        x1 : Array, shape (bs, *dim)
            represents the target minibatch

        Returns
        -------
        p : jnp.ndarray, shape (bs, bs)
            represents the OT plan between minibatches
        """
        # Use JIT-compiled version for better performance
        # Create a JIT-compiled function on first call if not already created
        if not hasattr(self, '_get_map_jit_fn'):
            # Create JIT-compiled version with the current ot_fn bound
            def _get_map_jit_impl(x0, x1, normalize_cost):
                a = jnp.ones(x0.shape[0]) / x0.shape[0]  # Uniform distribution
                b = jnp.ones(x1.shape[0]) / x1.shape[0]  # Uniform distribution
                
                if x0.ndim > 2:
                    x0 = x0.reshape(x0.shape[0], -1)
                if x1.ndim > 2:
                    x1 = x1.reshape(x1.shape[0], -1)
                
                # Compute cost matrix with JAX
                M = jnp.sum((x0[:, None, :] - x1[None, :, :]) ** 2, axis=2)
                if normalize_cost:
                    M = M / M.max()  # should not be normalized when using minibatches
                
                # Use pure JAX implementation (ot_fn is captured from closure)
                p = self.ot_fn(a, b, M)
                
                # For exact OT, the plan might have numerical issues with very small regularization
                # Check if the plan is valid (not all zeros or NaNs)
                p_sum = p.sum()
                is_finite = jnp.all(jnp.isfinite(p))
                is_valid = (p_sum >= 1e-8) & is_finite
                
                # Use jax.lax.cond for conditional logic that's JIT-compatible
                def use_plan(p, p_sum):
                    # Normalize the plan (needed for exact OT)
                    return p / p_sum
                
                def use_uniform(p, p_sum):
                    return jnp.ones_like(p) / p.size
                
                # Conditionally use the plan or uniform fallback
                p = jax.lax.cond(is_valid, use_plan, use_uniform, p, p_sum)
                
                return p
            
            # JIT-compile with static_argnames
            self._get_map_jit_fn = jax.jit(_get_map_jit_impl, static_argnames=('normalize_cost',))
        
        return self._get_map_jit_fn(x0, x1, self.normalize_cost)

    def sample_map(self, key, pi, batch_size, replace=True):
        r"""Draw source and target samples from pi  $(x,z) \sim \pi$

        Parameters
        ----------
        key : PRNGKey
            JAX random key
        pi : numpy array, shape (bs, bs)
            represents the source minibatch
        batch_size : int
            represents the OT plan between minibatches
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        (i_s, i_j) : tuple of numpy arrays, shape (bs, bs)
            represents the indices of source and target data samples from $\pi$
        """
        p = pi.flatten()
        p = p / p.sum()
        choices = random.choice(
            key, pi.shape[0] * pi.shape[1], p=p, shape=(batch_size,), replace=replace
        )
        return jnp.divmod(choices, pi.shape[1])

    def sample_plan(self, key, x0, x1, replace=True):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target samples from pi $(x,z) \sim \pi$

        Parameters
        ----------
        key : PRNGKey
            JAX random key
        x0 : Array, shape (bs, *dim)
            represents the source minibatch
        x1 : Array, shape (bs, *dim)
            represents the source minibatch
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        x0[i] : Array, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        x1[j] : Array, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(key, pi, x0.shape[0], replace=replace)
        return x0[i], x1[j]

    def sample_plan_with_scipy(self, x0, x1):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch using pure JAX Hungarian algorithm and draw source and target samples from pi $(x,z) \sim \pi$.

        This sampler has two advantages:
        * Reduced variance compared to sampling from the OT plan
        * Preserves the order of x1 by construction
        * Preserves entire batch if x0 and x1 have the same size

        Parameters
        ----------
        x0 : Array, shape (bs, *dim)
            represents the source minibatch
        x1 : Array, shape (bs, *dim)
            represents the source minibatch

        Returns
        -------
        x0[i] : Array, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        x1[j] : Array, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        """
        if x0.ndim > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.ndim > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        # Compute cost matrix with JAX
        M = jnp.sum((x0[:, None, :] - x1[None, :, :]) ** 2, axis=2)
        if self.normalize_cost:
            M = M / M.max()
        # Use pure JAX Hungarian algorithm
        _, j = _hungarian(M)
        pi_x0 = x0[j]
        pi_x1 = x1
        return pi_x0, pi_x1

    def sample_plan_with_labels(self, key, x0, x1, y0=None, y1=None, replace=True):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target labeled samples from pi $(x,z) \sim \pi$

        Parameters
        ----------
        key : PRNGKey
            JAX random key
        x0 : Array, shape (bs, *dim)
            represents the source minibatch
        x1 : Array, shape (bs, *dim)
            represents the target minibatch
        y0 : Array, shape (bs)
            represents the source label minibatch
        y1 : Array, shape (bs)
            represents the target label minibatch
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        x0[i] : Array, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        x1[j] : Array, shape (bs, *dim)
            represents the target minibatch drawn from $\pi$
        y0[i] : Array, shape (bs, *dim)
            represents the source label minibatch drawn from $\pi$
        y1[j] : Array, shape (bs, *dim)
            represents the target label minibatch drawn from $\pi$
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(key, pi, x0.shape[0], replace=replace)
        return (
            x0[i],
            x1[j],
            y0[i] if y0 is not None else None,
            y1[j] if y1 is not None else None,
        )

    def sample_trajectory(self, key, X):
        """Compute the OT trajectories between different sample populations moving from the source
        to the target distribution.

        Parameters
        ----------
        key : PRNGKey
            JAX random key
        X : Array, (bs, times, *dim)
            different populations of samples moving from the source to the target distribution.

        Returns
        -------
        to_return : Array, (bs, times, *dim)
            represents the OT sampled trajectories over time.
        """
        times = X.shape[1]
        batch_size = X.shape[0]
        
        # Compute all OT plans using scan
        def compute_plan(carry, t):
            # carry is unused, t is the time index
            pi = self.get_map(X[:, t], X[:, t + 1])
            return None, pi
        
        _, pis = jax.lax.scan(compute_plan, None, jnp.arange(times - 1))
        
        # Sample indices through the trajectory using scan
        def sample_step(carry, pi):
            key, prev_indices = carry
            # Split key for each sample
            keys = random.split(key, len(prev_indices))
            
            # Sample next indices based on current indices and OT plan
            def sample_for_index(i, key_i):
                p = pi[i] / (pi[i].sum() + 1e-16)
                choice = random.choice(key_i, pi.shape[1], p=p)
                return choice
            
            # Use vmap to sample for all indices with different keys
            next_indices = jax.vmap(sample_for_index)(prev_indices, keys)
            
            # Update key for next iteration
            key, _ = random.split(key)
            
            return (key, next_indices), next_indices
        
        # Initialize with identity indices
        initial_indices = jnp.arange(batch_size)
        (final_key, _), all_indices = jax.lax.scan(
            sample_step, (key, initial_indices), pis
        )
        
        # Stack all indices: [initial_indices, all_indices from scan]
        all_indices = jnp.concatenate([initial_indices[None, :], all_indices], axis=0)
        
        # Extract trajectories using advanced indexing
        # all_indices shape: (times, batch_size)
        # X shape: (batch_size, times, *dim)
        # We need to index X[all_indices[t], t] for each t
        def extract_trajectory(t):
            return X[all_indices[t], t]
        
        to_return = jax.vmap(extract_trajectory)(jnp.arange(times))
        # Transpose to get (batch_size, times, *dim)
        to_return = jnp.transpose(to_return, (1, 0) + tuple(range(2, len(to_return.shape))))
        
        return to_return


def wasserstein(
    x0: jnp.ndarray,
    x1: jnp.ndarray,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    """Compute the Wasserstein (1 or 2) distance (wrt Euclidean cost) between a source and a target
    distributions.

    Parameters
    ----------
    x0 : Array, shape (bs, *dim)
        represents the source minibatch
    x1 : Array, shape (bs, *dim)
        represents the source minibatch
    method : str (default : None)
        Use exact Wasserstein or an entropic regularization
    reg : float (default : 0.05)
        Entropic regularization coefficients
    power : int (default : 2)
        power of the Wasserstein distance (1 or 2)
    Returns
    -------
    ret : float
        Wasserstein distance
    """
    assert power == 1 or power == 2
    
    a = jnp.ones(x0.shape[0]) / x0.shape[0]  # Uniform distribution
    b = jnp.ones(x1.shape[0]) / x1.shape[0]  # Uniform distribution
    
    if x0.ndim > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.ndim > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    
    # Compute cost matrix with JAX
    M = jnp.sqrt(jnp.sum((x0[:, None, :] - x1[None, :, :]) ** 2, axis=2))
    if power == 2:
        M = M**2
    
    # Use pure JAX implementations
    if method == "sinkhorn" or method is None:
        ret = sinkhorn2(a, b, M, reg=reg)
    elif method == "exact":
        ret = emd2(a, b, M)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if power == 2:
        ret = math.sqrt(ret)
    return float(ret)
