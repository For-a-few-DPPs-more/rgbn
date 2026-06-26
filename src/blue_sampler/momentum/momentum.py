import numpy as np

from .momentum_clusters import (
    central_moment_orders,
    weighted_central_moments,
    solve_moments_lm,
)

from .momentum_polygons import central_moments as polygon_central_moments


def _target(distribution, geometry, orders, weights):

    if geometry == "clusters":

        B, N, _ = distribution.shape

        if weights is None:
            weights = np.full((B, N), 1 / N)

        return weighted_central_moments(
            distribution,
            weights,
            orders,
        )

    if geometry == "polygons":
        return polygon_central_moments(
            distribution,
            orders,
        )

    raise ValueError(geometry)


def _init(distribution, geometry, n, rng):

    B = distribution.shape[0]

    if geometry == "clusters":

        idx = np.argsort(
            rng.random(distribution.shape[:2]),
            axis=1,
        )[:, :n]

        return np.take_along_axis(
            distribution,
            idx[..., None],
            axis=1,
        )

    if geometry == "polygons":

        lo = distribution.min(1)
        hi = distribution.max(1)

        return rng.uniform(
            lo[:, None],
            hi[:, None],
            (B, n, 2),
        )

    raise ValueError(geometry)


def momentum_fit(
    distribution,
    distribution_type="clusters",
    n=6,
    p=3,
    weights=None,
    n_restarts=1,
    restart_tol=1e-8,
    n_iter=50,
    lambda0=1e-2,
    tol=1e-12,
    random_state=None,
):

    gtype = distribution_type
    D = distribution.shape[-1]

    if gtype == "polygons":
        assert D == 2
        assert p <= 3

    orders = central_moment_orders(p, D)

    target_centroid, target_moments = _target(
        distribution,
        gtype,
        orders,
        weights,
    )

    rng = np.random.default_rng(random_state)

    best_points = None
    best_cost = None
    history = None

    for _ in range(max(1, n_restarts)):

        init = _init(
            distribution,
            gtype,
            n,
            rng,
        )

        points, history, cost = solve_moments_lm(
            target_centroid,
            target_moments,
            orders,
            init,
            n_iter=n_iter,
            lambda0=lambda0,
            tol=tol,
        )

        if best_points is None:

            best_points = points
            best_cost = cost

        else:

            better = cost < best_cost

            best_points = np.where(
                better[:, None, None],
                points,
                best_points,
            )

            best_cost = np.where(
                better,
                cost,
                best_cost,
            )

        if np.all(best_cost < restart_tol):
            break

    return best_points, history