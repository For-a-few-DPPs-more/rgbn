"""
Public API.

`blue.sample_points` is the main entry point of the package.
It provides a high-level interface for generating large point sets on the
periodic unit hypercube [0, 1)^D with sub-Poisson density fluctuations
(so-called blue noise).

The package also exposes utilities to sample tessellations (2D only) and
balanced clusters (arbitrary dimension). 
tessels or clusters are sampled with following balance property: 
uniform area repartition if no target is given or uniform atoms repartition if
target atoms are given.

Clusters and tessellations can subsequently be converted into low-discrepancy
point sets using `blue.from_geometry' which internally solves a 
moment-matching problem
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .run.run_bruteforce import _bruteforce_pipeline
from .run.run_recursive import _recursive_pipeline
from .warm_start import _sobol_warmstart
from .progress import ProgressLogger

from .run.run_tessels import _tesselation
from .run.run_clusters import _clusterisation

from .momentum.momentum import momentum_fit

from .viz import plot

_PRESETS = {
    2: dict(spatial_radius=7, spectral_radius=7, LR_spatial=0.100, LR_spectral=0.1, expension_factor=0.3, S=1.0),
    3: dict(spatial_radius=5, spectral_radius=5, LR_spatial=0.030, LR_spectral=0.1, expension_factor=0.3, S=1.0),
    4: dict(spatial_radius=3, spectral_radius=3, LR_spatial=0.010, LR_spectral=0.1, expension_factor=1.0, S=0.5),
    5: dict(spatial_radius=3, spectral_radius=3, LR_spatial=0.003, LR_spectral=0.1, expension_factor=1.5, S=0.5),
}


def im2points(image = "anything.jpg", N = 100_000):
    """
    simple wrapper for image stippling
    """
    sample = sample_points(N = N, D = 2, targets = image)
    plot(sample, figsize = (10, 10))
    return sample

def sample_points(
    N: int = 2**15,
    D: int = 2,
    targets: NDArray | None = None,
    bruteforce: bool = False,
    warmstart: str | NDArray | None = None,
    n_iter: int = 6,
    verbose: int = 1,
) -> NDArray:
    """
    Generate N stealthy (blue-noise) points in [0, 1)^D.

    Parameters
    ----------
    N : int
        Number of output points.
    D : int
        Spatial dimension.
    bruteforce : bool, default False
        Use the bruteforce O(N^2) algorithm instead of the recursive one.
        Gives a better sample but is intractable for N >= 50_000.
        Automatically forced to True when N <= 2_000.
    warmstart : {None, "Sobol", ndarray of shape (N, D)}, default None
        Initial point configuration.
        - None : default random/recursive initialisation.
        - "Sobol" : initialise with a Sobol low-discrepancy sequence
          (requires scipy.stats.qmc.Sobol).
        - ndarray : use given points as the starting configuration.
    n_iter : int, default 6
        Number of solver iterations. Each iteration runs 10 gradient
        steps plus a structural gridification step (neighbor lookup).
        More iterations = better quality but slower.
    targets : ndarray of shape (K, D), optional
        Atoms describing a target density.
        can also be a path to an image, e.g. targets = "zebra.jpg"
  
    verbose : int, default 1
        0 = silent, 1 = live progress.

    Returns
    -------
    points : ndarray of shape (N, D)
        The sampled point coordinates in [0, 1)^D.
    """
    has_target = targets is not None
    if has_target:
        n_iter *= 2

    if isinstance(warmstart, np.ndarray):
        if warmstart.shape != (N, D):
            raise ValueError(f"warmstart must have shape {(N, D)}, got {warmstart.shape}")
        x = warmstart.copy()
    elif warmstart is None:
        x = None
    elif warmstart == "Sobol":
        x = _sobol_warmstart(N, D)
    else:
        raise ValueError(f"unsupported warmstart={warmstart!r}, expected None, a custom np.array or 'Sobol'")

    use_bruteforce = bruteforce or N <= 2_000

    if verbose >= 1:
        print(f"✦ {D}D blue-noise pipeline — sampling {N:,} points")

    if n_iter == 0:
        return x

    logger = ProgressLogger(D, verbose)
    if use_bruteforce:
        ctx = logger.enter_level(N, D, 0)
        ctx.start()
        blue = _bruteforce_pipeline(
            N, D, n_iter, ctx,
            target=targets,
        )
        sampled_points  = np.array(blue(x))
        logger.exit_level()
    
    else:
        preset = _PRESETS[min(D, 5)]
        sampled_points = _recursive_pipeline(
            N=N,
            D=D,
            N_ITER=n_iter,
            logger=logger,
            S=preset["S"],
            expension_factor=preset["expension_factor"],
            LR_spatial=preset["LR_spatial"],
            LR_spectral=preset["LR_spectral"],
            spatial_radius=preset["spatial_radius"],
            spectral_radius=preset["spectral_radius"],
            N_PER_STEP=10,
            x=x,
            target=targets,
        )

    if verbose >= 1:
        print("Done - To see result: blue.plot(x)")

    return sampled_points

def sample_tessels(
    N: int = 2**15,
    D: int = 2,
    targets: NDArray | None = None,
    return_atoms: bool = False,
) -> NDArray | tuple[NDArray, NDArray]:
    """
    Recursively split the unit square into N random quadrilaterals.

    If `targets` is None, splits are computed to achieve equal areas.
    If `targets` is provided, splits are computed to achieve a median
    separation of the atoms.

    Parameters
    ----------
    N : int, default 1024
        Number of final tessels. Must be a power of 2.
    targets : ndarray of shape (K, 2), optional
        Coordinates of atoms to split, in the [0, 1)^2 unit box.
        K must be a multiple of N. The more targets provided, the better
        the approximation, but the slower the computation
        (K/N should be at least 100 for a decent tessellation).
        A typical use case is adaptive tessellation, e.g. with `targets`
        being i.i.d. points sampled from a target density.

    Returns
    -------
    quad : ndarray of shape (N, 4, 2)
        A tessellation composed of N quadrilaterals with equal area
        or equal number of atoms.
    atoms : ndarray of shape (N, K/N, 2)
        The input target atoms, redistributed among their final quadrilateral.
        Only returned if `targets` was provided and return_atoms is set to True.

    Notes
    -----
    Only supports 2D geometry and a power-of-two number of tessels (N).
    """
    assert D ==2, f"{D}D tesselation is currently unsupported so sample_tessels requires D = 2"
    depth = int(np.log2(N))
    assert 2**depth == N, "N must be a power of 2 because at each step each tessel is splitted to produce 2 "
    has_targets = targets is not None
    if has_targets:
        if targets.ndim == 2:
            targets = targets[None, ...]
        assert targets.shape[1] % N == 0, f"The number of targets ({targets.shape[1]}) must be a multiple of N ({N})."

    if (return_atoms == True) or (targets is None):
        return _tesselation(depth, targets)
    return _tesselation(depth, targets)[0]

def sample_clusters(
    N: int = 2**15,
    D: int = 2,
    targets: NDArray | None = None,
    n_per_cluster: int = 16,
) -> NDArray:
    """
    Recursively partition a point set into N balanced clusters.

    At each recursion step, every cluster is split into two equal halves
    using a random median hyperplane. After log2(N) recursion levels,
    exactly N clusters are obtained.

    Parameters
    ----------
    N : int, default 1024
        Number of final clusters. Must be a power of two.

    D : int, default 2
        Ambient dimension.

    targets : ndarray of shape (K, D), optional
        Initial atoms to clusterise.
        If omitted, a Sobol low-discrepancy sequence containing
        K = N * n_per_cluster atoms is generated automatically.

    n_per_cluster : int, default 16
        Number of atoms per final cluster.
        Only used when `targets` is not provided.

    Returns
    -------
    ndarray of shape (N, K/N, D)
        Collection of balanced clusters.

    Notes
    -----
    The recursive splitting procedure requires the total number of atoms
    K to be divisible by N.
    """

    depth = int(np.log2(N))

    if (1 << depth) != N:
        raise ValueError(
            f"N={N} must be a power of two."
        )

    if targets is not None:

        if targets.ndim == 2:
            targets = targets[None, :, :]

        K = targets.shape[1]

        if K % N != 0:
            raise ValueError(
                f"The number of targets ({K}) "
                f"must be divisible by N ({N})."
            )
        
    return _clusterisation(
        depth=depth,
        D=D,
        targets=targets,
        n_per_cluster=n_per_cluster,
    )

def tile(x: NDArray, repeat: int, flatoutput: bool = True) -> NDArray:
    """
    Tile points on the unit torus to cover [0, 1)^D periodically.

    Each of the `repeat**D` copies of `x` is rescaled by `1/repeat` and
    shifted to its own sub-cube, so that the copies together pave the
    unit torus again. For example in 2D with
    repeat=2: tile (0, 0) holds x/2, tile (1, 1) holds x/2 + 0.5, etc.

    Parameters
    ----------
    x : ndarray, shape (N, D)
        Points in [0, 1)^D (unit torus).
    repeat : int
        Number of repetitions per axis. The output therefore contains
        Nfinal = N * repeat**D points.
    flatoutput : bool, default True
        If True, reshape the output to (Nfinal, D). If False, keep the
        tile structure as leading axes.

    Returns
    -------
    ndarray, shape (Nfinal, D) if flatoutput else (repeat, ..., repeat, N, D)
        Tiled version of `x`, periodized over [0, 1)^D.
    """
    N, D = x.shape

    grids = np.meshgrid(*([np.arange(repeat)] * D), indexing="ij")
    idx = np.stack(grids, axis=-1)                            # (repeat,)*D + (D,)
    offset = (idx / repeat).reshape(*([repeat] * D), 1, D)    # (repeat,)*D + (1, D)

    x_scaled = x / repeat                                      # (N, D)
    xtiled = x_scaled + offset                                 # (repeat,)*D + (N, D)

    return xtiled.reshape(-1, D) if flatoutput else xtiled

def from_geometry(
    geometry: NDArray,
    gtype: str,
    n: int = 6,
    p: int = 3,
    **kwargs
) -> NDArray:
    """
    Convert tessels or clusters into low-discrepancy point set.

    Each tessel or cluster is replaced by n points by solving a
    moment-matching problem (Levenberg-Marquardt) via `momentum_fit`.

    Parameters
    ----------
    geometry : ndarray
        Batch of tessels or clusters to convert.
        - gtype="polygons" : quadrilaterals of shape (N, 4, 2),
          e.g. the `quad` output of `sample_tessels`.
        - gtype="clusters" : point sets of shape (N, K, D),
          e.g. the output of `sample_clusters`, or the `atoms` output of
          `sample_tessels(..., return_atoms=True)`.
    gtype : {"polygons", "clusters"}
        Geometry type. "polygons" only supports D = 2.
    n : int, default 6
        Number of output points per tessel or cluster.
    p : int, default 3
        Maximum total moment order to match (centroid plus central moments
        up to order p). geometry_type="polygons" only supports p <= 3.
    **kwargs :
        Additional arguments passed to `momentum_fit` (e.g., n_restarts, 
        random_state, tol).

    Returns
    -------
    ndarray of shape (N, n, D)
        n points per tessel or cluster, matching its moments up to order p.
    """
    if gtype not in ("clusters", "polygons"):
        raise ValueError(
            f"Unknown geometry type: {gtype!r}, "
            f"expected one of ('clusters', 'polygons')"
        )
        
    if gtype == "polygons" and p > 3:
        raise ValueError("Exact polygon moments only support order p <= 3.")


    points, _ = momentum_fit(
        distribution=geometry,
        distribution_type=gtype,
        n=n,
        p=p,
        **kwargs
    )
    
    return points