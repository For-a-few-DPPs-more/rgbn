import warnings
from math import gcd
from typing import Literal

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Generating vectors based on optimised Korobov sequences (Rank-1 Lattices)
#
# The generating vector is z = (1, a, a^2, ..., a^{D-1}) mod N.
# The integer 'a' is chosen to maximise the spectral test (the length of the
# shortest non-zero vector in the dual lattice). This ensures near-optimal
# equidistribution and minimises the worst-case integration error for smooth
# periodic functions on the torus.
#
# Instead of an exhaustive grid search (which scales exponentially with D),
# this implementation uses a simplified Lenstra-Lenstra-Lovász (LLL) lattice
# reduction algorithm to evaluate candidates in O(D^3) time. This makes it
# highly efficient and scalable even for N = 1,000,000 and D <= 6.
# ---------------------------------------------------------------------------

class _KorobovOptimizer:
    """
    Internal optimizer for finding the best Korobov generating vector 'a'
    using LLL lattice reduction to approximate the spectral test.
    """
    def __init__(self, N: int, D: int):
        self.N = int(N)
        self.D = int(D)

    def generator(self, a: int) -> NDArray:
        """Compute z = (1, a, a^2, ..., a^{D-1}) mod N."""
        z = np.empty(self.D, dtype=np.int64)
        z[0] = 1
        for j in range(1, self.D):
            z[j] = (z[j - 1] * a) % self.N
        return z

    def get_dual_basis(self, a: int) -> NDArray:
        """
        Construct an explicit basis for the dual lattice L^perp.
        Condition: h · z ≡ 0 (mod N).
        The resulting D x D matrix has determinant exactly N.
        """
        z = self.generator(a)
        B = np.zeros((self.D, self.D), dtype=np.int64)
        B[0, 0] = self.N
        for i in range(1, self.D):
            B[i, 0] = (-z[i]) % self.N
            B[i, i] = 1
        return B

    @staticmethod
    def lll_reduce(B: NDArray, delta: float = 0.99) -> NDArray:
        """
        Simplified, standalone LLL reduction.
        Finds a short vector in the lattice spanned by the rows of B.
        For D <= 6, the first vector is virtually always the true shortest
        vector (lambda_1), with an approximation guarantee of 2^{(D-1)/4} <= 2.37.
        """
        B = B.astype(np.float64).copy()
        D = B.shape[0]
        k = 1

        def gso():
            B_star = np.zeros_like(B)
            mu = np.zeros((D, D))
            for i in range(D):
                B_star[i] = B[i].copy()
                for j in range(i):
                    mu[i, j] = np.dot(B[i], B_star[j]) / np.dot(B_star[j], B_star[j])
                    B_star[i] -= mu[i, j] * B_star[j]
            return B_star, mu

        B_star, mu = gso()
        
        while k < D:
            # Size reduction
            for j in range(k - 1, -1, -1):
                q = round(mu[k, j])
                if q != 0:
                    B[k] -= q * B[j]
            
            # Recompute GSO
            B_star, mu = gso()
            
            # Lovász condition
            norm_k = np.dot(B_star[k], B_star[k])
            norm_k_1 = np.dot(B_star[k-1], B_star[k-1])
            
            if norm_k >= (delta - mu[k, k-1]**2) * norm_k_1:
                k += 1
            else:
                # Swap and step back
                B[[k, k-1]] = B[[k-1, k]]
                B_star, mu = gso()
                k = max(1, k - 1)
                
        return np.round(B).astype(np.int64)

    def find_best_a(self, max_candidates: int | None = None) -> tuple[int, NDArray]:
        """
        Search for the generator 'a' coprime to N that maximises the 
        shortest dual vector length (lambda_1).
        """
        candidates = [a for a in range(2, self.N) if gcd(a, self.N) == 1]
        
        if max_candidates is not None and len(candidates) > max_candidates:
            # Deterministic random sampling for reproducibility of the search
            rng = np.random.default_rng(42)
            candidates = rng.choice(candidates, size=max_candidates, replace=False)
            candidates = sorted(candidates)

        best_a = 1
        best_lambda1 = -1.0
        best_z = self.generator(1)

        for a in candidates:
            B = self.get_dual_basis(a)
            B_reduced = self.lll_reduce(B)
            lambda1 = float(np.linalg.norm(B_reduced[0].astype(np.float64)))
            
            if lambda1 > best_lambda1:
                best_lambda1 = lambda1
                best_a = a
                best_z = self.generator(a)
                
        return best_a, best_z


# ---------------------------------------------------------------------------
# Public warm-start samplers
# ---------------------------------------------------------------------------

def _goodlattice_warmstart(
    N: int,
    D: int,
    seed: int | None = None,
    shift: bool = True,
    max_candidates: int | None = 5000,
) -> NDArray:
    """
    Generate N points in [0, 1)^D using an optimised Korobov rank-1 lattice
    with an optional random toroidal shift.

    The generating vector z = (1, a, a^2, ..., a^{D-1}) mod N is chosen
    to maximise the spectral test. The point set is defined by:

        x_k = (seed_shift + k * z / N) mod 1,   k = 0, ..., N-1

    Instead of an exhaustive grid search, this implementation uses a simplified
    Lenstra-Lenstra-Lovász (LLL) lattice reduction algorithm to evaluate 
    candidates in O(D^3) time, making it highly scalable.

    The toroidal shift (randomised quasi-Monte Carlo / RQMC) preserves the
    low-discrepancy structure while breaking the deterministic starting
    point, making estimates unbiased and enabling variance estimation
    across multiple replicates.

    Parameters
    ----------
    N : int
        Number of points.
    D : int
        Ambient dimension. Best results and LLL guarantees apply for D <= 6.
    seed : int or None, optional
        Seed for the RNG used to draw the toroidal shift.
        If None, a random shift is drawn non-reproducibly.
        Ignored when shift=False.
    shift : bool, optional
        Whether to apply a random toroidal shift (RQMC).
        Set to False to obtain the fully deterministic sequence.
        Default: True.
    max_candidates : int or None, optional
        Maximum number of coprime candidates 'a' to evaluate.
        If None, all coprime integers in [2, N-1] are checked.
        For large N (e.g., 1,000,000), a limit of 2000-5000 is recommended
        to keep execution time under a few seconds while still finding
        near-optimal generators. Default: 5000.

    Returns
    -------
    sample : ndarray of shape (N, D)
        Lattice points in [0, 1)^D.

    Notes
    -----
    The LLL reduction provides a 2^{(D-1)/4} approximation guarantee for the
    shortest vector. For D <= 6, this factor is <= 2.37, meaning the first
    vector of the reduced basis is virtually always the true shortest vector
    (lambda_1) of the dual lattice. This yields a star discrepancy of 
    O((log N)^D / N), the best achievable rate for rank-1 constructions.

    References
    ----------
    Sloan, I. H., & Joe, S. (1994). Lattice Methods for Multiple Integration.
    Oxford University Press.

    Niederreiter, H. (1992). Random Number Generation and Quasi-Monte Carlo
    Methods. SIAM.
    """
    optimizer = _KorobovOptimizer(N, D)
    _, z = optimizer.find_best_a(max_candidates=max_candidates)

    # Convert integer generator to fractional step sizes in [0, 1)
    z_float = z.astype(np.float64) / N

    k = np.arange(N, dtype=np.float64)
    points = (k[:, None] * z_float) % 1.0

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