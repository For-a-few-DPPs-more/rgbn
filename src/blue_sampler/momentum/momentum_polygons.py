"""Moment matching for simple polygons in R^2.

Notation follows Lotz & Klatt, "Persistence of asymptotic variance under
transport" (arXiv:2605.22803), Eq. (1.8); see momentum_clusters.py for the
shared docstring on q-indexing and central-moment conventions. Here the
"cell" C is a polygon (vertices ordered along its boundary) and the
quadrature nodes X_1, ..., X_n are matched against the polygon's own
moments 1/lambda_2(C) * int_C x^q dx instead of a discrete sum.

Raw moments int_P x^i y^j dA of a simple polygon P are computed via
Green's theorem as a sum over edges (no hardcoded order cutoff, valid for
any i, j >= 0):

    int_P x^i y^j dA = 1/(i+j+2) * sum_edges cross_e * int_0^1 x_e(t)^i y_e(t)^j dt

where, for an edge from (x, y) to (xn, yn): cross_e = x*yn - xn*y,
x_e(t) = x + t*(xn - x), y_e(t) = y + t*(yn - y). The inner integral has
the closed binomial form used in `_edge_integral` below.
"""

from math import comb

import numpy as np


def _edge_integral(i, j, x, xn, y, yn):
    """int_0^1 x(t)^i y(t)^j dt for x(t) = x + t*(xn - x), y(t) = y + t*(yn - y).

    Closed binomial-expansion form (verified against direct symbolic
    integration): sum_{a<=i, b<=j} C(i,a) C(j,b) x^a dx^{i-a} y^b dy^{j-b}
    / ((i-a)+(j-b)+1).
    """

    dx = xn - x
    dy = yn - y

    out = np.zeros_like(x)

    for a in range(i + 1):

        xa = x ** a * dx ** (i - a)

        for b in range(j + 1):

            denom = (i - a) + (j - b) + 1

            out += (
                comb(i, a) * comb(j, b)
                * xa
                * y ** b * dy ** (j - b)
                / denom
            )

    return out


def raw_moments(vertices, order):
    """Raw moments {(i, j): M_ij} for 0 <= i + j <= order, via Green's theorem."""

    assert vertices.shape[-1] == 2

    x = vertices[..., 0]
    y = vertices[..., 1]

    xn = np.roll(x, -1, axis=1)
    yn = np.roll(y, -1, axis=1)

    cross = x * yn - xn * y

    out = {
        (0, 0): 0.5 * cross.sum(1)
    }

    for total in range(1, order + 1):
        for i in range(total + 1):

            j = total - i

            integral = _edge_integral(i, j, x, xn, y, yn)

            out[(i, j)] = np.sum(
                cross * integral,
                axis=1,
            ) / (i + j + 2)

    return out


def central_moments(vertices, orders):
    """Centroid and central moments of a polygon, for the given (i, j) orders.

    `orders` may be empty (e.g. p = 2, centroid only): the centroid is
    always computed from raw moments up to order 1, and `moments` is
    returned with shape (B, 0).
    """

    assert vertices.shape[-1] == 2

    max_order = max((sum(alpha) for alpha in orders), default=1)

    raw = raw_moments(
        vertices,
        max(max_order, 1),
    )

    area = raw[(0, 0)]

    cx = raw[(1, 0)] / area
    cy = raw[(0, 1)] / area

    centroid = np.stack(
        [cx, cy],
        axis=1,
    )

    moments = np.empty(
        (vertices.shape[0], len(orders))
    )

    for k, (i, j) in enumerate(orders):

        m = np.zeros_like(area)

        for a in range(i + 1):
            for b in range(j + 1):

                m += (
                    comb(i, a)
                    * comb(j, b)
                    * (-cx) ** (i - a)
                    * (-cy) ** (j - b)
                    * raw[(a, b)]
                )

        moments[:, k] = m / area

    return centroid, moments