import numpy as np

def _sobol_warmstart(N: int, D: int, seed: int | None = None) -> np.ndarray:
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
        raise ImportError("Install scipy first to use Sobol sampling")

    engine = qmc.Sobol(d=D, seed=seed)
    sample = engine.random(N)
    return sample