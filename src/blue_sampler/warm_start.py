import warnings
from math import gcd
from typing import Literal

import numpy as np
from numpy.typing import NDArray



# ---------------------------------------------------------------------------
# Generating vectors based on generalised golden ratios (R-sequence / Kronecker)
#
# The d-th component is phi_d = 1 / alpha_d where alpha_d is the unique
# positive root of x^(d+1) = x + 1.  These constants minimise the
# worst-case L2 discrepancy for rank-1 lattices on the torus and are the
# most irrational numbers in their respective dimensions.
#
# Reference: Roberts (2018) "The Unreasonable Effectiveness of Quasirandom
# Sequences", http://extremelearning.com.au/unreasonable-effectiveness-of-
# quasirandom-sequences/
# ---------------------------------------------------------------------------

def _phi(d: int) -> float:
    """
    Compute the d-th generalised golden ratio constant alpha_d.

    alpha_d is the unique positive root > 1 of  x^(d+1) = x + 1.
    Newton's method converges in ~20 iterations for any d <= 100.
    """
    x = 1.0 + 1.0 / d          # initial guess (close to 1 for large d)
    for _ in range(64):
        fx  =  x ** (d + 1) - x - 1
        dfx = (d + 1) * x ** d - 1
        x  -= fx / dfx
    return x


def _r_sequence_vector(D: int) -> NDArray:
    """
    Build the D-dimensional generating vector for the R-sequence.

    z[d] = 1 / alpha_{d+1}  (the fractional part used as step size).
    The resulting Kronecker lattice  x_k = k * z mod 1  achieves
    near-optimal equidistribution on the torus for any N.

    Returns
    -------
    z : ndarray of shape (D,), dtype float64, values in (0, 1).
    """
    return np.array([1.0 / _phi(d + 1) for d in range(D)])


# ---------------------------------------------------------------------------
# Public warm-start samplers
# ---------------------------------------------------------------------------

def _kronecker_warmstart(
    N: int,
    D: int,
    seed: int | None = None,
    shift: bool = True,
) -> NDArray:
    """
    Generate N points in [0, 1)^D using the R-sequence (generalised
    golden-ratio Kronecker lattice) with an optional random toroidal shift.

    The point set is defined by

        x_k = (seed_shift + k * z) mod 1,   k = 0, ..., N-1

    where z[d] = 1 / alpha_{d+1} and alpha_{d+1} is the unique root > 1
    of  t^(d+2) = t + 1.  These constants are the most irrational numbers
    in their respective dimensions and minimise the worst-case L2
    discrepancy for rank-1 lattices on the unit torus.

    The toroidal shift (randomised quasi-Monte Carlo / RQMC) preserves the
    low-discrepancy structure while breaking the deterministic starting
    point, making estimates unbiased and enabling variance estimation
    across multiple replicates.

    Parameters
    ----------
    N : int
        Number of points.
    D : int
        Ambient dimension.
    seed : int or None, optional
        Seed for the RNG used to draw the toroidal shift.
        If None, a random shift is drawn non-reproducibly.
        Ignored when shift=False.
    shift : bool, optional
        Whether to apply a random toroidal shift (RQMC).
        Set to False to obtain the fully deterministic sequence.
        Default: True.

    Returns
    -------
    sample : ndarray of shape (N, D)
        Lattice points in [0, 1)^D.

    Notes
    -----
    Compared to drawing z randomly among coprime integers, the golden-ratio
    vector is provably optimal and requires no candidate search or
    discrepancy evaluation.  The resulting lattice has O((log N)^D / N)
    star discrepancy, the best achievable rate for rank-1 constructions.

    References
    ----------
    Roberts, M. (2018). "The Unreasonable Effectiveness of Quasirandom
    Sequences." http://extremelearning.com.au/unreasonable-effectiveness-of-
    quasirandom-sequences/

    Dick, J., Kuo, F. Y., & Sloan, I. H. (2013). High-dimensional
    integration: the quasi-Monte Carlo way. Acta Numerica, 22, 133-288.
    """
    z = _r_sequence_vector(D)           # shape (D,), deterministic

    k = np.arange(N, dtype=np.float64)  # shape (N,)
    points = np.outer(k, z) % 1.0       # shape (N, D)

    if shift:
        rng = np.random.default_rng(seed)
        delta = rng.random(D)           # uniform toroidal shift in [0, 1)^D
        points = (points + delta) % 1.0

    return points


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