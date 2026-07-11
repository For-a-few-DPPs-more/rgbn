from __future__ import annotations

import itertools
import time
from functools import partial

import math

import numpy as np
from numpy.typing import NDArray
import jax
import jax.numpy as jnp
import optax
from squarenet import SquareNet

from ..math import torus_delta, torus_wrap


def pad_fit_transform(x: NDArray, sn: SquareNet) -> NDArray:
    """
    SquareNet can wright arbitrary points as a grid.
    here we need to pad points first so that they are 
    as many points as grid cells. Empty cells are filed with NaN.
    """
    x = np.array(x)  
    x = x[np.isfinite(x).all(axis = -1)]
    D = x.shape[-1]
    x = x.reshape(-1, D)
    x_paded = np.random.rand(sn.N, D)
    x_paded[:len(x)] = x
    order = np.random.permutation(np.arange(sn.N))
    sn.fit(x_paded[order], method="ultimate")
    if len(x) < sn.N:
        x_paded[len(x):] = np.nan
    x_new = sn.map(x_paded[order])
    return x_new #(G, ..., G, D)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_shifts(radius: int, dims: int) -> jnp.ndarray:
    """
    Return all integer lattice shifts inside the L2 ball of given radius.

    Parameters
    ----------
    radius : int
        Ball radius (inclusive).
    dims : int
        Number of dimensions.

    Returns
    -------
    jnp.ndarray
        Shape ``(S, dims)`` integer array of shift vectors.
    """
    ranges = [range(-radius, radius + 1)] * dims
    shifts = [
        s for s in itertools.product(*ranges)
        if sum(x ** 2 for x in s) <= radius ** 2
    ]
    return jnp.array(shifts, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def _nufft_pipeline(
    N: int,
    D: int,
    lr: float = 1.0,
    kfrac: float = 0.3,
    warmstart: NDArray | None = None,
    verbose: int = 1,
    n_iter: int = 300,
) -> NDArray:
    """Generate a low-discrepancy point set in $[0, 1)^D$ via 
    NUFFT (= non uniform fast fourier) spectral energy minimisation.

    Parameters
    ----------
    N : int
        Number of points to generate.
    D : int
        Dimension of the space.
    lr : float, default=1.0
        Learning rate scaling multiplier.
    kfrac : float, default=0.3
        Frequency cut-off fraction.
        Energy is minimised up to wavevectors K s.t. ||K|| = kfrac*Kmax.
    warmstart : NDArray, optional
        Initial point coordinates of shape `(N, D)`. Default is random.
    verbose : bool, default=True
        If True, logs optimization progress.
    n_iter : int, default=300
        Number of gradient descent iterations.

    Returns
    -------
    NDArray
        Optimised point coordinates array of shape `(N, D)`.
    """
    G = math.ceil(N ** (1.0 / D))
    true_G = N ** (1.0/D)

    # ------------------------------------------------------------------
    # Logging helper
    # ------------------------------------------------------------------
    def log(msg: str) -> None:
        if verbose >=1:
            print(msg)

    # ------------------------------------------------------------------
    # Hyper-parameters & Grid setup
    # ------------------------------------------------------------------
    K_RADIUS = max(1, int(true_G*kfrac))
    SIGMA2 = (1.0 / G) ** 2
    learning_rate = (0.1 / G * lr) 

    # Coordinate grid in [0, 1)^D (dynamically scaled for D dimensions)
    g_axis = jnp.linspace(0.0, 1.0, G, endpoint=False)
    mesh = jnp.meshgrid(*([g_axis] * D), indexing="ij")
    GRID = jnp.stack(mesh, axis=-1)

    # Frequency mask and weights
    freqs = jnp.fft.fftfreq(G) * G
    freq_mesh = jnp.meshgrid(*([freqs] * D), indexing="ij")
    freq_r2 = sum(f ** 2 for f in freq_mesh)
    MASK = (freq_r2 <= K_RADIUS ** 2) & (freq_r2 > 0.0)

    # Avoid division by zero if MASK is empty (e.g., very small G)
    min_freq = freq_r2[MASK].min() if jnp.any(MASK) else 1.0
    K_WEIGHT = 1.0 / (freq_r2 + 0.01 * min_freq)

    # Dynamic FFT setup over spatial axes
    AXES = tuple(range(D))
    
    def fft(x):
        return jnp.fft.fftn(x, axes=AXES)
        
    def ifft(x):
        return jnp.fft.ifftn(x, axes=AXES)
    
    def nan_to_0(x):
        return jnp.nan_to_num(x)

    # ------------------------------------------------------------------
    # Loss + gradient
    # ------------------------------------------------------------------
    def _loss_and_grad(
        x_grid: jnp.ndarray,
        shifts: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward + backward pass for the D-Dimensional spectral energy."""

        def _forward(acc, shift):
            rolled = jnp.roll(x_grid, shift, axis=AXES)
            delta = torus_delta(GRID - rolled)
            dist2 = jnp.sum(delta ** 2, axis=-1)
            return acc + nan_to_0(jnp.exp(-dist2 / SIGMA2)), None

        density, _ = jax.lax.scan(_forward, jnp.zeros(tuple([G] * D)), shifts)

        F = fft(density)
        amps = (jnp.abs(F) ** 3) * K_WEIGHT
        
        mask_sum = jnp.maximum(MASK.sum(), 1.0) # Avoid division by zero
        loss = jnp.sum(jnp.where(MASK, amps, 0.0)) / mask_sum

        grad_F = (3.0 / mask_sum) * MASK * K_WEIGHT * jnp.abs(F) * F
        
        # Generalised scaling factor: G**D instead of hardcoded G**2
        grad_density = (G ** D) * ifft(grad_F).real

        def _backward(acc_grad, shift):
            rolled = jnp.roll(x_grid, shift, axis=AXES)
            delta = torus_delta(GRID - rolled)
            dist2 = jnp.sum(delta ** 2, axis=-1, keepdims=True)
            local_exp = jnp.exp(-dist2 / SIGMA2)
            grad_pixel = grad_density[..., None] * (2.0 / SIGMA2) * nan_to_0(local_exp * delta)
            return acc_grad + jnp.roll(grad_pixel, -shift, axis=AXES), None

        grad_x, _ = jax.lax.scan(_backward, jnp.zeros_like(x_grid), shifts)
        return loss, grad_x

    # ------------------------------------------------------------------
    # JIT-compiled optimisation loop
    # ------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(1,2))
    def _run_optimization(
        x_init: jnp.ndarray,
        n_steps: int,
        learning_rate: float,
        shifts: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        optimizer = optax.adam(learning_rate)

        def _step(carry, _):
            x, opt_state = carry
            loss, grads = _loss_and_grad(x, shifts)
            updates, opt_state = optimizer.update(grads, opt_state)
            x_new = torus_wrap(x + updates)
            return (x_new, opt_state), loss

        opt_state = optimizer.init(x_init)
        (x_final, _), losses = jax.lax.scan(
            _step, (x_init, opt_state), None, length=n_steps
        )
        return x_final, losses
    
    # ------------------------------------------------------------------
    # Shift kernels (adaptative heuristics per dimension)
    # ------------------------------------------------------------------
    if D == 2:
      r_warmup = 10
      r_final  = 7
    if D == 3:
      r_warmup = 5
      r_final  = 5
    if D == 4:
      r_warmup = 4
      r_final  = 4
    if D >= 5:
      r_warmup = 3
      r_final  = 3
    
    shifts_warmup = get_shifts(r_warmup, D)
    shifts_final  = get_shifts(r_final, D)

    # ------------------------------------------------------------------
    # Initial points
    # ------------------------------------------------------------------
    if warmstart is not None:
        if warmstart.shape != (N, D):
            raise ValueError(
                f"warmstart must have shape ({N}, {D}), got {warmstart.shape}."
            )
        x_np = warmstart.astype(np.float32)
    else:
        x_np = np.random.default_rng(0).random((N, D)).astype(np.float32)

    gridshape = tuple([G] * D)
    sn = SquareNet(gridshape=gridshape, backend="jax", verbose=0, max_iter = 300)

    # ------------------------------------------------------------------
    # Stage 1 – initial gridification
    # ------------------------------------------------------------------
    log(f"[nufft_sampling] {D}D | grid {'×'.join([str(G)] * D)} | {N} points")
    log("[1/4] Initial gridification…")
    t0 = time.time()
    x_grid = pad_fit_transform(x_np, sn)
    log(f"      done in {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # Stage 2 – warmup
    # ------------------------------------------------------------------
    log("[2/4] Warmup (30 steps)…")
    t0 = time.time()
    x_grid, losses_warmup = _run_optimization(x_grid, n_iter//2, learning_rate, shifts_warmup)
    losses_warmup.block_until_ready()
    log(
        f"      done in {time.time() - t0:.2f}s | "
        f"loss {losses_warmup[0]:.4f} → {losses_warmup[-1]:.4f}"
    )

    # ------------------------------------------------------------------
    # Stage 3 – re-gridification
    # ------------------------------------------------------------------
    log("[3/4] Re-gridification…")
    t0 = time.time()
    pts_flat = np.array(x_grid).reshape(-1, D)
    x_grid = pad_fit_transform(pts_flat, sn)
    log(f"      done in {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # Stage 4 – final optimisation
    # ------------------------------------------------------------------
    log(f"[4/4] Final optimisation ({n_iter} steps)…")
    t0 = time.time()
    x_final, losses_final = _run_optimization(x_grid, n_iter, 0.2*learning_rate, shifts_final)
    x_final.block_until_ready()
    log(
        f"      done in {time.time() - t0:.2f}s | "
        f"loss {losses_final[0]:.4f} → {losses_final[-1]:.4f}"
    )
    x_final = np.array(x_final).reshape(-1, D)
    return x_final[np.isfinite(x_final).all(axis = -1)]