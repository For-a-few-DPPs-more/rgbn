import math

import numpy as np


FORMULAS = {
    (1, 0): lambda x, xn, y, yn: (x + xn) / 6,
    (0, 1): lambda x, xn, y, yn: (y + yn) / 6,
    (2, 0): lambda x, xn, y, yn: (x**2 + x*xn + xn**2) / 12,
    (1, 1): lambda x, xn, y, yn: (2*x*y + x*yn + xn*y + 2*xn*yn) / 24,
    (0, 2): lambda x, xn, y, yn: (y**2 + y*yn + yn**2) / 12,
    (3, 0): lambda x, xn, y, yn: (x**3 + x**2*xn + x*xn**2 + xn**3) / 20,
    (2, 1): lambda x, xn, y, yn:
        (3*x**2*y + x**2*yn + 2*x*xn*y + 2*x*xn*yn + xn**2*y + 3*xn**2*yn)/60,
    (1, 2): lambda x, xn, y, yn:
        (3*x*y**2 + y**2*xn + 2*x*y*yn + 2*xn*y*yn + x*yn**2 + 3*xn*yn**2)/60,
    (0, 3): lambda x, xn, y, yn: (y**3 + y**2*yn + y*yn**2 + yn**3)/20,
}


def raw_moments(vertices, order):

    assert vertices.shape[-1] == 2
    assert order <= 3

    x = vertices[..., 0]
    y = vertices[..., 1]

    xn = np.roll(x, -1, axis=1)
    yn = np.roll(y, -1, axis=1)

    cross = x * yn - xn * y

    out = {
        (0, 0): 0.5 * cross.sum(1)
    }

    for key, f in FORMULAS.items():

        if sum(key) > order:
            continue

        out[key] = np.sum(
            cross * f(x, xn, y, yn),
            axis=1,
        )

    return out


def central_moments(vertices, orders):

    assert vertices.shape[-1] == 2

    raw = raw_moments(
        vertices,
        max(sum(o) for o in orders),
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
                    math.comb(i, a)
                    * math.comb(j, b)
                    * (-cx) ** (i - a)
                    * (-cy) ** (j - b)
                    * raw[(a, b)]
                )

        moments[:, k] = m / area

    return centroid, moments