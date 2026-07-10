import numpy as np
from numpy.typing import NDArray

_BASE = np.array([[0.,0.],[2.,0.],[0.,1.]])

def step_transform():
    return [
        (np.array([[-1,-2],[-2, 1]])/5,  np.array([2/5, 4/5])),  # T1
        (np.array([[ 2,-1],[-1,-2]])/5,  np.array([1/5, 2/5])),  # T2
        (np.array([[ 2, 1],[-1, 2]])/5,  np.array([1/5, 2/5])),  # T3
        (np.array([[ 2,-1],[-1,-2]])/5,  np.array([6/5, 2/5])),  # T4
        (np.array([[-2,-1],[ 1,-2]])/5,  np.array([6/5, 2/5])),  # T5 
    ]

def _subdivide(triangles):
    """(N,3,2) → (5N,3,2)"""
    result = []
    for M, t in step_transform():
        result.append(np.einsum('ij,nkj->nki', M, triangles) + t)
    return np.concatenate(result, axis=0)

def _full_transform(pinwheel_base, tiling):
    pinwheel_base   = np.asarray(pinwheel_base,   dtype=float)
    tiling = np.asarray(tiling, dtype=float)
    if pinwheel_base.ndim == 2:
        pinwheel_base = pinwheel_base[None]          # (1,3,2)

    Q0, Q1, Q2 = pinwheel_base[0, 0], pinwheel_base[0, 1], pinwheel_base[0, 2]       # (2,)
    P0, P1, P2 = tiling[:, 0], tiling[:, 1], tiling[:, 2]  # (N,2)

    A     = np.column_stack([Q1 - Q0, Q2 - Q0])   # (2,2)  
    B     = np.stack([P1 - P0, P2 - P0], axis=2)  # (N,2,2)

    M = B @ np.linalg.inv(A)                       # (N,2,2)
    t = P0 - (M @ Q0[:, None])[:, :, 0]     # (N,2)

    return M, t