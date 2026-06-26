from itertools import product

import numpy as np


def central_moment_orders(p, D):

    out = [
        a
        for a in product(range(p + 1), repeat=D)
        if 2 <= sum(a) <= p
    ]

    return sorted(
        out,
        key=lambda a: (sum(a), a),
    )


def weighted_central_moments(points, weights, orders):

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

    B, N, D = points.shape

    centroid = np.sum(
        points * weights[None, :, None],
        axis=1,
    )

    delta = points - centroid[:, None]

    Jc = np.zeros((B, D, N, D))

    for d in range(D):
        Jc[:, d, :, d] = weights

    Jc = Jc.reshape(B, D, N * D)

    moments = np.empty((B, len(orders)))
    Jm = np.zeros((B, len(orders), N * D))

    for k, alpha in enumerate(orders):

        term = np.ones((B, N))

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
    n_iter=50,
    lambda0=1e-2,
    tol=1e-12,
):

    B, N, D = init_points.shape

    w = np.full(N, 1 / N)

    points = init_points.copy()

    eye = np.eye(N * D)[None]

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

    for _ in range(n_iter):

        JT = J.transpose(0, 2, 1)

        A = JT @ J + lam[:, None, None] * eye

        g = (JT @ r[..., None])[..., 0]

        delta = np.linalg.solve(
            A,
            -g[..., None],
        )[..., 0]

        trial = points.reshape(B, -1) + delta
        trial = trial.reshape(B, N, D)

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

    return points, history, cost