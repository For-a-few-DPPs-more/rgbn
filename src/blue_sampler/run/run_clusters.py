""" 
Balanced recursive clusterisation in arbitrary dimension D. 

Unlike the 2D version (see run_tessels.py), there is no longer 
any notion of a geometric polygon. The state consists solely of
a set of target atoms (latent clusters), organized into batches. 

At each recursion step, every batch is split into two equal halves 
using a median separation along a random direction drawn uniformly 
from the unit sphere S^(D-1). 

Atoms can be sampled from either a (possibly non homogeneous) 
distribution given by the user, or from a sobol sequence (default). 
"""

from ..warm_start import _sobol_warmstart

import numpy as np

# ------------------------------------------------------------
# Balanced split using a random median hyperplane
# ------------------------------------------------------------

def fair_random_split(targets, rng):
    """
    Split each batch of points into two equal halves using a median
    partition along a random direction.

    A different random direction is sampled independently for each batch,
    uniformly from the unit sphere S^(D-1).

    Parameters
    ----------
    targets : ndarray of shape (B, K, D)

        Input batches of points. K must be even.

    rng : np.random.Generator

        Random number generator.

    Returns
    -------
    ndarray of shape (2*B, K//2, D)

        The first B batches correspond to the "lower" halves and the next
        B batches correspond to the "upper" halves, following the same
        stacking convention as the original 2D implementation.
    """

    B, K, D = targets.shape

    if K % 2 != 0:
        raise ValueError(
            f"K={K} must be even in order to split into two equal halves"
        )

    half = K // 2

    # One random direction per batch, uniformly distributed on S^(D-1).

    directions = rng.normal(size=(B, D)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Scalar projection of each point onto its batch direction:
    #
    #     h[b, k] = <targets[b, k], directions[b]>
    #

    h = np.einsum("bkd,bd->bk", targets, directions)

    # Sort points by projection value and split at the median.

    sort_idx = np.argsort(h, axis=-1)

    targets_sorted = np.take_along_axis(
        targets,
        sort_idx[..., None],
        axis=1,
    )

    targets_low = targets_sorted[:, :half, :]
    targets_high = targets_sorted[:, half:, :]

    return np.concatenate(
        (targets_low, targets_high),
        axis=0,
    )

# ------------------------------------------------------------

# Main pipeline

# ------------------------------------------------------------

def _clusterisation(
    depth,
    D,
    targets=None,
    n_per_cluster=16,
    rng=None,
    ):
    """
    Complete clusterisation pipeline for arbitrary dimension D.

    ```
    Workflow
    --------

    1. Generate an initial set of atoms.

    By default a Sobol low-discrepancy sequence is used.

    2. Apply recursive balanced median splits.

    Each split partitions every batch into two equal halves along
    a randomly oriented hyperplane.

    3. Repeat for ``depth`` levels.

    The final result contains ``2**depth`` balanced clusters.

    Parameters
    ----------
    depth : int

        Recursion depth.

    D : int

        Ambient dimension.

    targets : ndarray, optional

        Custom initial point cloud of shape (1, K, D).

        If omitted, a Sobol sequence is generated automatically.

    n_per_cluster : int, default=100

        Desired number of atoms per final cluster when generating the
        Sobol initialization.

    rng : np.random.Generator, optional

        Random number generator.

    Returns
    -------
    ndarray of shape (N, n_per_cluster, D)

        Collection of balanced clusters, where

            N = 2**depth.
    """

    rng = np.random.default_rng() if rng is None else rng

    N = 1 << depth

    if targets is None:

        K = N * n_per_cluster

        targets = _sobol_warmstart(K, D)

        if targets.ndim == 2:
            targets = targets[None, :, :]

    else:

        K = targets.shape[1]

        if K % N != 0:
            raise ValueError(
                f"K={K} must be divisible by N={N}"
            )

    targets = targets.astype(np.float32)

    for _ in range(depth):
        targets = fair_random_split(targets, rng)

    return targets

def back_merge_clusters(clusters):
    """
    Inverse operation of a recursion level.

    ```
    Merge sibling clusters back together following the ordering
    convention used by ``fair_random_split``.
    """

    chalf = len(clusters) // 2

    return np.concatenate(
        (clusters[:chalf], clusters[chalf:]),
        axis=1,
    )
