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
from typing import Literal
from .run.run_bruteforce import _bruteforce_pipeline
from .run.run_recursive import _PRESETS, _recursive_pipeline
from .run.run_nufft import _nufft_pipeline
from .warm_start import _sobol_warmstart, _x_warmstart
from .progress import ProgressLogger

from .run.run_tessels import _tesselation
from .run.run_clusters import _clusterisation

from .pinwheels import _BASE, _subdivide, _full_transform

from .momentum.momentum import _from_geometry

from .viz import plot

def im2points(image = "anything.jpg", N = 100_000):
    """
    simple wrapper for image stippling

    Return
    ------
    points : ndarray of shape (N, D)
        The sampled point coordinates in [0, 1)^D.
    """
    points = sample_points(N = N, D = 2, targets = image)
    plot(points, figsize = (10, 10))
    return points

def sample_points(
    N: int = 2**15,
    D: int = 2,
    lr: float = 1.0,
    method: Literal["rgbn", "nufft", "bruteforce"] = "rgbn",
    warmstart: Literal["Sobol", "Pinwheel"] | NDArray = None, 
    n_iter: int = 6,
    targets: NDArray | None = None,
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
    lr: learning rate (prefactor), used to scale internal learning rate
    method : {"rgbn", "nufft", "bruteforce"}, default "rgbn"
        Sampling method.
        - "nufft": spectral loss non-uniform fast Fourier transform.
        - "rgbn": spatial loss (truncated) based on recursive Gaussian Blue-Noise.
        - "bruteforce":  exact GBN (no truncation, better but much slower).
    warmstart : {None, "Sobol", ndarray of shape (N, D)}, default None
        Initial point configuration.
        - None : default random/recursive initialisation.
        - "Sobol" : initialise with a Sobol low-discrepancy sequence
          (requires scipy.stats.qmc.Sobol).
        - "Pinwheel" : initialise with the pinwheel method
            (only for 2D), see pinwheel_transform
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

    Note
    ----
    bruteforce is automaticaly used for N <= 2_000 
    as in that regime it is the best method from any
    perspective.
    """
    methods = ["rgbn", "bruteforce", "nufft"]
    if not method in methods:
        raise ValueError(f"unknown method {method}, must be one of {methods}")
    bruteforce = method == "bruteforce" or N <= 2_000
    nufft = method == "nufft"

    has_target = targets is not None
    if has_target:
        if method == "nufft":
            raise ValueError("a target was given but method nufft does'nt support a custom target")
        n_iter *= 2

    if warmstart is None:
        x = None
    else: 
        lr /= 2
        n_iter *= 2
        if isinstance(warmstart, np.ndarray):
            if warmstart.shape != (N, D):
                raise ValueError(f"warmstart must have shape {(N, D)}, got {warmstart.shape}")
            x = warmstart.copy()
        elif warmstart == "Sobol":
            x = _sobol_warmstart(N, D)
        elif warmstart == "Pinwheel":
            if D == 2:
                x = _pinwheel_warmstart(N)
            else:
                print("[warmstart] pinwheel asked but D > 2...\n -> fallback to Sobol")
        else:
            raise ValueError(f"unsupported warmstart={warmstart!r}, expected None, a custom np.array, 'Sobol' or 'Pinwheel'")
    
    if nufft:
        return _nufft_pipeline(N, D, lr = lr, warmstart = warmstart, 
                               verbose = verbose, n_iter = 50*n_iter)

    if verbose >= 1:
        print(f"✦ {D}D blue-noise pipeline — sampling {N:,} points")

    if n_iter == 0:
        return x

    logger = ProgressLogger(D, verbose)
    if bruteforce:
        ctx = logger.enter_level(N, D, 0)
        ctx.start()
        blue = _bruteforce_pipeline(
            N, D, n_iter, ctx = ctx,
            lr = lr,
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
            LR_spatial=lr * preset["LR_spatial"],
            LR_spectral=lr * preset["LR_spectral"],
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

def cluster2points(clusters: NDArray, p: int = 3) -> NDArray:
    """
    Convert clusters into low-discrepancy point set.

    Each cluster is replaced by m points by solving a
    moment-matching problem (Levenberg-Marquardt) via `momentum_fit`.

    Parameters
    ----------
    clusters: NDArray of shape (N, k, d)
         e.g. the output of `sample_clusters`, or the `atoms` output of
          `sample_tessels(..., return_atoms=True)`.
    p : int, default 3
        Maximum total moment order to match (centroid plus central moments
        up to order p). 
    Returns
    -------
    ndarray of shape (N, m, D)
        m points per tessel or cluster, matching its moments up to order p.
    """
    return _from_geometry(clusters, "clusters", p)

def tessel2points(tessels: NDArray, p: int = 3) -> NDArray:
    """
    Convert tessels into low-discrepancy point set.

    Each tessel is replaced by m points by solving a
    moment-matching problem (Levenberg-Marquardt) via `momentum_fit`.

    Parameters
    ----------
    tessels: NDArray of shape (n, 4, 2)
          e.g. the `quad` output of `sample_tessels`.
    p : int, default 3
        Maximum total moment order to match (centroid plus central moments
        up to order p). 
    **kwargs :
        Additional arguments passed to `momentum_fit` (e.g., n_restarts, 
        random_state, tol).

    Returns
    -------
    ndarray of shape (N, m, 2)
        m points per tessel or cluster, matching its moments up to order p.
    """
    return _from_geometry(tessels, "polygons", p)

def sobol(N:int = 2**15, D:int = 2):
    """
    Simple wrapper upon scipy.stats.qcm.sobol.

    Parameters
    ----------
    N : int
        Number of output points.
    D : int
        Spatial dimension.
    
    Returns
    -------
    points : ndarray of shape (N, D)
        The sampled point coordinates in [0, 1)^D
        from a scrambled Sobol sequence

    Notes
    -----
    It is recomanded (but not mandatory) that N is a power of 2
    for higher accuracy
    """
    return _sobol_warmstart(N= N, D = D)

def pinwheel_base():
    """
    Returns the base Conway triangle for pinwheel aperiodic tiling
    """
    return _BASE.copy()

def pinwheel_transform(
    points: NDArray = pinwheel_base(), 
    depth: int = 4, 
) -> NDArray:
    """
    Apply a Pinwheel tiling transformation to a set of points.

    This function generates a fractal tiling pinwheel_based on the recursive subdivision 
    of the Conway triangle (Pinwheel tiling) and maps the input points onto 
    each of the resulting triangles.

    Parameters
    ----------
    points : ndarray, default is the pinwheel_base triangle
        The coordinates of the points to transform, living on the pinwheel_base triangle.
        - If shape is (2,): the same point is projected onto every 
          triangle in the tiling.
        - If shape is (M, 2): the same set of points is projected onto every 
          triangle in the tiling.
        - If shape is (N, M, 2): each set of M points is projected onto its 
          corresponding triangle N.
    depth : int, optional
        Number of subdivision iterations. The number of resulting triangles 
        grows exponentially with depth. Default is 5.

    Returns
    -------
    ndarray
        An array of shape (N, M, 2) containing the transformed points for 
        each of the N triangles generated at the specified depth.

    Notes
    -----
    The Pinwheel tiling is a non-periodic tiling where each triangle is 
    subdivided into 5 smaller triangles, each rotated by an angle of 
    $\arctan(1/2)$ relative to the parent.
    """
    tiling = pinwheel_base()[None]
    for _ in range(depth):  
        tiling = _subdivide(tiling)

    tiling = np.concatenate([tiling, -tiling + np.array([[2.0, 1.0]])], axis = 0)
    tiling = np.concatenate([tiling, tiling* np.array([[-1.0, 1.0]]) + np.array([[2.0, 1.0]])], axis = 0)
    M, t = _full_transform(pinwheel_base(), tiling/2.0)

    if points.ndim == 1:
        points = np.einsum('nij,j-> ni', M, points) + t
    elif points.ndim == 2:
        points = np.einsum('nij,kj->nki', M, points) + t[:, None, :]
    else:
        points = np.einsum('nij,nkj->nki', M, points) + t[:, None, :]
    return points

def _pinwheel_warmstart(N):
    """
    simple wrapper upon tessel2points + pinwheel_transform to sample exactly N points
    only for 2D
    """
    xbase = tessel2points(pinwheel_base(), p = 3) #(3, 2)
    depth = int(np.log(N/3)/np.log(5) + 1)
    intensity = (3*4*5**depth)
    x = pinwheel_transform(xbase, depth = depth) #(4*depth**5, 3, 2)
    return _x_warmstart(x, N, intensity = intensity) #(N, 2)