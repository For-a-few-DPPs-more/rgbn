import warnings
from math import gcd
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def _kronecker_warmstart(
    N: int,
    D: int,
    seed: int | None = None,
) -> NDArray:
    """
    Generate N points in [0, 1)^D using a rank-1 Kronecker lattice.

    The generated point set is a periodic lattice on the unit torus:
    x_k = k * z / N mod 1, where z is an integer generating vector.
    Choosing the components of z coprime with N ensures that the full
    period contains exactly N distinct points.

    Kronecker lattices provide deterministic, highly uniform designs and
    are often useful as warm starts for optimization, clustering,
    tessellation, or geometric sampling algorithms.

    Parameters
    ----------
    N : int
        Number of points.

    D : int
        Ambient dimension.

    seed : int or None, optional
        Seed for the random generator used to sample the generating vector.

    Returns
    -------
    sample : np.ndarray of shape (N, D)
        Kronecker lattice points in [0, 1)^D.

    Notes
    -----
    The generating vector z is sampled randomly among integer vectors whose
    components are coprime with N.

    Compared to i.i.d. uniform sampling, Kronecker lattices have lower
    discrepancy and preserve periodicity, making them suitable for
    algorithms operating on the torus.
    """
    rng = np.random.default_rng(seed)

    z = np.empty(D, dtype=int)

    for d in range(D):
        while True:
            candidate = int(rng.integers(1, N))
            if gcd(candidate, N) == 1:
                z[d] = candidate
                break

    k = np.arange(N)[:, None]
    return (k * z[None, :] / N) % 1.0


def _sobol_warmstart(
    N: int,
    D: int,
    seed: int | None = None,
) -> NDArray:
    """
    Generate N points in [0, 1)^D using a Sobol low-discrepancy sequence.

    Sobol sequences provide a deterministic space-filling design with much
    lower discrepancy than independent uniform random sampling.

    Parameters
    ----------
    N : int
        Number of points.

    D : int
        Ambient dimension.

    seed : int or None, optional
        Seed passed to SciPy's Sobol generator.

    Returns
    -------
    sample : np.ndarray of shape (N, D)
        Sobol points in [0, 1)^D.

    Notes
    -----
    This function is a lightweight wrapper around
    `scipy.stats.qmc.Sobol`.
    """
    try:
        from scipy.stats import qmc
    except ImportError as exc:
        raise ImportError(
            "Install scipy first to use Sobol warm starts"
        ) from exc

    engine = qmc.Sobol(d=D, seed=seed)
    return engine.random(N)


def _x_warmstart(
    x: NDArray,
    N: int,
    intensity: float | None = None,
) -> NDArray:
    """
    Extract exactly N points from an existing point cloud and map them to
    the unit torus [0, 1)^D.

    The procedure assumes that a representative square/cubic subsample can
    be extracted from x. Unexpected results may occur otherwise.

    Parameters
    ----------
    x : ndarray
        Input point cloud.

    N : int
        Number of points to extract.

    intensity : float or None, optional
        Theoretical intensity of the point cloud. If None, it is estimated.

    Returns
    -------
    sample : np.ndarray of shape (N, D)
        Point cloud mapped to [0, 1)^D.
    """
    rng = np.random.default_rng()

    D = x.shape[-1]

    rotation, _ = np.linalg.qr(rng.normal(size=(D, D)))

    x = x.reshape(-1, D) @ rotation
    x = x - x.mean(axis=0, keepdims=True)

    xnorm = np.max(np.abs(x), axis=1)

    order = np.argsort(xnorm)
    x = x[order]
    xnorm = xnorm[order]

    if intensity is None:
        intensity = ((2 * xnorm[-1]) ** D) / len(x)

    L = (N / intensity) ** (1 / D)

    x = x[:N] / L + 0.5

    return x - np.floor(x)