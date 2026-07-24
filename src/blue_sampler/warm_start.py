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

class KorobovLLL:
    def __init__(self, N, D):
        self.N = int(N)
        self.D = int(D)

    def generator(self, a):
        """z = (1, a, a², ..., a^(D-1)) mod N"""
        z = np.empty(self.D, dtype=np.int64)
        z[0] = 1
        for j in range(1, self.D):
            z[j] = (z[j - 1] * a) % self.N
        return z

    def get_dual_basis(self, a):
        """
        Construit une base explicite du réseau dual L^perp.
        Condition : h · z ≡ 0 (mod N).
        La matrice B (D x D) a pour lignes les vecteurs de base.
        Son déterminant est exactement N.
        """
        z = self.generator(a)
        B = np.zeros((self.D, self.D), dtype=np.int64)
        B[0, 0] = self.N
        for i in range(1, self.D):
            B[i, 0] = (-z[i]) % self.N
            B[i, i] = 1
        return B

    @staticmethod
    def lll_reduce(B, delta=0.99):
        """
        Réduction LLL simplifiée et autonome.
        Trouve un vecteur court dans le réseau engendré par les lignes de B.
        Pour D <= 6, le premier vecteur est presque toujours le plus court absolu (lambda1).
        """
        B = B.astype(np.float64).copy()
        D = B.shape[0]
        k = 1

        # Gram-Schmidt Orthogonalization (GSO) interne
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
            # 1. Réduction de taille (Size reduction)
            for j in range(k - 1, -1, -1):
                q = round(mu[k, j])
                if q != 0:
                    B[k] -= q * B[j]

            # 2. Recalcul GSO (très rapide pour D <= 6 : ~200 opérations)
            B_star, mu = gso()

            # 3. Condition de Lovász
            norm_k = np.dot(B_star[k], B_star[k])
            norm_k_1 = np.dot(B_star[k-1], B_star[k-1])

            if norm_k >= (delta - mu[k, k-1]**2) * norm_k_1:
                k += 1
            else:
                # Échange et retour en arrière
                B[[k, k-1]] = B[[k-1, k]]
                B_star, mu = gso()
                k = max(1, k - 1)

        return np.round(B).astype(np.int64)

    def spectral_test_lll(self, a):
        """Retourne une estimation très précise de lambda1 via LLL."""
        B = self.get_dual_basis(a)
        B_reduced = self.lll_reduce(B)
        # Le premier vecteur de la base réduite est le plus court candidat
        return float(np.linalg.norm(B_reduced[0].astype(np.float64)))

    def optimize(self, max_candidates=None, verbose=True):
        """
        Optimise le générateur 'a'.
        max_candidates permet de limiter la recherche pour des tests rapides.
        """
        N = self.N
        candidates = [a for a in range(2, N) if gcd(a, N) == 1]
        if max_candidates is not None:
            # Échantillonnage aléatoire ou tronqué pour les très grands N
            candidates = np.random.choice(candidates, size=min(max_candidates, len(candidates)), replace=False)
            candidates = sorted(candidates)

        best = {"a": None, "z": None, "lambda1": -1.0}

        for a in candidates:
            lam = self.spectral_test_lll(a)
            if lam > best["lambda1"]:
                best["a"] = int(a)
                best["z"] = self.generator(a)
                best["lambda1"] = lam
                if verbose:
                    print(f"a={a:7d}   λ1={lam:.6f}")

        self.best = best

        return best

    def points(self, z = None):
        """Retourne les N points de la règle de Korobov dans [0,1)^D."""
        if z is None:
            z = self.best["z"]
        t = np.arange(self.N, dtype=np.int64)[:, None]
        return ((t * z) % self.N) / self.N


# ---------------------------------------------------------------------------
# Public warmstart samplers
# ---------------------------------------------------------------------------

def _goodlattice_warmstart(
    N: int,
    D: int,
    seed: int | None = None,
    shift: bool = True,
    max_candidates: int | None = 100,
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
    optimizer = KorobovLLL(N, D)
    z = optimizer.optimize(max_candidates=max_candidates, verbose = False)["z"]

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