"""Moment matching for point clusters (weighted point sets in R^D).

Notation follows Lotz & Klatt, "Persistence of asymptotic variance under
transport" (arXiv:2605.22803), Eq. (1.8):

    1/lambda_d(C) * int_C x^q dx = 1/n * sum_{j=1}^n X_j^q,   q in {0, ..., p-1}

Here a "cell" C is replaced by a weighted point cloud (the `distribution`
to be matched), q ranges over multi-indices alpha with |alpha| = sum(alpha)
in {0, ..., p-1}, and x^q denotes the monomial x_1^{alpha_1} ... x_D^{alpha_D}.

q = 0 is the trivial total-mass condition (always satisfied once weights
sum to 1). q = 1 fixes the centroid. q >= 2 are genuine *central* moment
constraints once the centroid has been subtracted out, which is how they
are represented here (`moment_orders` only returns alpha with
2 <= |alpha| <= p - 1; the centroid itself is handled separately).
"""

from itertools import product

import numpy as np


def moment_orders(p, D):
    """Multi-indices alpha in N_0^D with 2 <= |alpha| <= p - 1.

    These are the non-trivial central-moment orders q referenced in
    Eq. (1.8) for q in {0, ..., p-1}: q = 0 (mass) and q = 1 (centroid)
    are excluded since they are handled separately.
    """

    out = [
        alpha
        for alpha in product(range(p), repeat=D)
        if 2 <= sum(alpha) <= p - 1
    ]

    return sorted(
        out,
        key=lambda alpha: (sum(alpha), alpha),
    )


def weighted_central_moments(points, weights, orders):
    """Centroid and central moments of a weighted point cloud.

    points:  (B, n, D)   -- X_1, ..., X_n per batch
    weights: (B, n)       -- normalized weights (sum to 1 along axis 1)
    orders:  list of multi-indices alpha, |alpha| >= 2

    Returns (centroid, moments) with centroid: (B, D), moments: (B, len(orders)).
    """

    centroid = np.sum(
        points * weights[..., None],
        axis=1,
    )

    delta = points - centroid[:, None]

    B = points.shape[0]

    moments = np.empty((B, len(orders)))

    for k, alpha in enumerate(orders):

        term = np.ones(delta.shape[:2])

        for d, a in enumerate(alpha):
            term *= delta[..., d] ** a

        moments[:, k] = np.sum(
            weights * term,
            axis=1,
        )

    return centroid, moments


def moments_and_jacobian(points, weights, orders):
    """Centroid, central moments, and their Jacobians w.r.t. point coordinates.

    weights are treated as fixed (1/n each); only point positions X_j vary.
    """

    B, n, D = points.shape

    centroid = np.sum(
        points * weights[None, :, None],
        axis=1,
    )

    delta = points - centroid[:, None]

    Jc = np.zeros((B, D, n, D))

    for d in range(D):
        Jc[:, d, :, d] = weights

    Jc = Jc.reshape(B, D, n * D)

    moments = np.empty((B, len(orders)))
    Jm = np.zeros((B, len(orders), n * D))

    for k, alpha in enumerate(orders):

        term = np.ones((B, n))

        for d, a in enumerate(alpha):
            term *= delta[..., d] ** a

        moments[:, k] = np.sum(
            weights * term,
            axis=1,
        )

        for d, a in enumerate(alpha):

            if a == 0:
                continue

            deriv = a * delta[..., d] ** (a - 1)

            for j, b in enumerate(alpha):

                if j != d:
                    deriv *= delta[..., j] ** b

            mean = np.sum(
                weights * deriv,
                axis=1,
                keepdims=True,
            )

            Jm[:, k, d::D] = (
                weights
                * (deriv - mean)
            )

    return centroid, moments, Jc, Jm


def solve_moments_lm(
    target_centroid,
    target_moments,
    orders,
    init_points,
    verbose=1,
    n_iter=50,
    lambda0=1e-2,
    tol=1e-12,
):
    """Levenberg-Marquardt solve for X_1, ..., X_n matching centroid + moments.

    Weights are fixed to 1/n (equal-weight quadrature / averaging set, as in
    Eq. (1.8): "1/n * sum_j X_j^q").
    """

    B, n, D = init_points.shape

    w = np.full(n, 1 / n)

    points = init_points.copy()

    eye = np.eye(n * D)[None]

    lam = np.full(B, lambda0)

    history = []

    def residuals(x):

        c, m, Jc, Jm = moments_and_jacobian(
            x,
            w,
            orders,
        )

        r = np.concatenate(
            [
                c - target_centroid,
                m - target_moments,
            ],
            axis=1,
        )

        J = np.concatenate(
            [Jc, Jm],
            axis=1,
        )

        return r, J

    r, J = residuals(points)

    cost = 0.5 * np.sum(r * r, axis=1)

    if verbose >= 1:
        import time
        print(f"✦ {D}D Moment matching pipeline — sampling {B:,} x {n:,} points\n")
        t0 = time.perf_counter()


    for it in range(n_iter):
        if verbose >= 1:
            if it <= 3:
                eta = "calibrating…"
            if it == 4:
                dt = time.perf_counter() - t0
                eta = _duration(dt/4 * (n_iter - 1))
            width = 20
            done = ((it+1) * width) // n_iter
            bar = "▓" * (done) + "░" * (width - done)
            print(
                f"\rMatching target… [{bar}] "
                f"{it+1}/{n_iter}  ETA ",
                eta.ljust(len("calibrating…")),
                end="",
                flush=True,
            )

        JT = J.transpose(0, 2, 1)

        JTJ = JT @ J

        trace_JTJ = np.trace(JTJ, axis1=1, axis2=2)  # shape (B,)
        min_lam = trace_JTJ * 1e-7/JTJ.shape[1]      # safety
        lam = np.maximum(lam, min_lam)

        A = JTJ + lam[:, None, None] * eye

        g = (JT @ r[..., None])[..., 0]

        delta = np.linalg.solve(
            A,
            -g[..., None],
        )[..., 0]

        trial = points.reshape(B, -1) + delta
        trial = trial.reshape(B, n, D)

        r2, J2 = residuals(trial)

        cost2 = 0.5 * np.sum(
            r2 * r2,
            axis=1,
        )

        ok = cost2 < cost

        points = np.where(
            ok[:, None, None],
            trial,
            points,
        )

        r = np.where(
            ok[:, None],
            r2,
            r,
        )

        J = np.where(
            ok[:, None, None],
            J2,
            J,
        )

        cost = np.where(
            ok,
            cost2,
            cost,
        )

        lam = np.where(
            ok,
            lam * 0.5,
            lam * 3.0,
        )

        history.append(cost.max())

        if cost.max() < tol:
            break

    if verbose >= 1:
        bar = "▓" * (width)
        print(
            f"\rMatching target… [{bar}] "
            f"{n_iter}/{n_iter}  ETA ",
            eta.ljust(len("calibrating…")),
            end="",
            flush=True,
        )
        time.sleep(0.05)
        print("\nDone")

    return points, history, cost

def _duration(seconds: float) -> str:
        seconds = round(seconds)
        if seconds < 60:
            return f"{seconds:.0f}s"
        minutes, rest = divmod(seconds, 60)
        return f"{minutes:.0f}min{rest:02.0f}"