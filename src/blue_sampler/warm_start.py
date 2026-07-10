import numpy as np
from numpy.typing import NDArray

def _sobol_warmstart(N: int, D: int, seed: int | None = None) -> NDArray:
    """
    Generate N points in [0, 1)^D using a Sobol low-discrepancy sequence.

    Sobol sequences provide a deterministic space-filling design with much
    lower discrepancy than independent uniform random sampling. They are
    often useful as an initialization ("warm start") for optimization,
    clustering, tessellation, or density estimation algorithms.

    Parameters
    ----------
    N : int
        Number of points to generate.

    D : int
        Ambient dimension.

    seed : int or None, optional
        Seed passed to SciPy's Sobol generator. Using the same seed
        reproduces the same sequence.

    Returns
    -------
    sample : np.ndarray of shape (N, D)
        Sobol points in the unit hypercube [0, 1)^D.

    Notes
    -----
    This function is a lightweight wrapper around
    `scipy.stats.qmc.Sobol`.

    Compared to i.i.d. uniform samples, Sobol points cover the domain
    more evenly and typically provide a better initialization for
    geometric algorithms.
    """
    try:
        from scipy.stats import qmc
    except ImportError:
        raise ImportError("Install scipy first to use warm starts")

    engine = qmc.Sobol(d=D, seed=seed)
    sample = engine.random(N)
    return sample

def _x_warmstart(x: NDArray, N: int, intensity: float = None) -> NDArray:
    """
    helper to get exactly N points in the unit square [0, 1)^D from 
    a sample x with more than N points and not necessary on the unit square,
    not even on a square at all. It might fail or produce unexpected results
    if one can't extract a square subsample of x, user should check that.
    User must know the theoreticall intensity of x, other wise it will estimated
    and estimate is NOT precise
    """
    D  = x.shape[-1]
    random_rotation, _      = np.linalg.qr(np.random.randn(D, D)) 
    x = x.reshape(-1, D) @ random_rotation
    x = x - x.mean(axis = 0, keepdims = True)
    xnorm = np.max(np.abs(x), axis = 1)
    order = np.argsort(xnorm)
    x = x[order]
    xnorm = xnorm[order]
    if intensity is None:
        intensity = ((2*xnorm)**D)/N
    L = (N/intensity)**(1/D)
    x = x[:N]/L + 0.5
    return x - np.floor(x)
