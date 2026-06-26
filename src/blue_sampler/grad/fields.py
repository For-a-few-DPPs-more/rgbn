"""
Fields allow to control the density of the points for adaptative sampling
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from ..math import clean_grad
from .im2fields import im2field

def _fourier_grad_kernels(shape, sigma2_kernel, D):
    shape = np.asarray(shape)
    freqs = [np.fft.fftfreq(shape[d], d=1.0 / shape[d]) for d in range(D)]
    mesh = np.meshgrid(*freqs, indexing="ij")
    C = np.prod(shape) * (np.pi * sigma2_kernel) ** (D / 2.0)
    f2 = sum(mesh[d] ** 2 for d in range(D))
    V_hat = C * np.exp(-(np.pi ** 2) * sigma2_kernel * f2)
    grad_hat = [
        1j * 2.0 * np.pi * mesh[d] * V_hat
        for d in range(D)
    ]
    return grad_hat, V_hat

def _log_barier(rho):
    eps = 1e-4
    alpha = rho.mean()*1e-3
    log_barier = alpha/((rho/rho.mean()).clip(min = 0) + eps)*(1+np.random.rand(*rho.shape))
    return log_barier

def _density_from_points(target, shape):
    D = target.shape[1]
    idx = np.floor(target * shape).astype(np.int64) % shape
    rho = np.zeros(tuple(shape))
    np.add.at(rho, tuple(idx[:, d] for d in range(D)), 1.0 / len(target))
    return rho

def _field_from_density(rho, shape, sigma2_kernel, D):
    rho = rho 
    anti_rho = rho.max() - rho
    field = np.empty(tuple(shape) + (D,))
    grad_hat, V_hat = _fourier_grad_kernels(shape, sigma2_kernel, D)
    rho_smooth = np.real(np.fft.ifftn(np.fft.fftn(rho)*V_hat))
    anti_rho_hat = np.fft.fftn(anti_rho - _log_barier(rho_smooth)) 

    for d in range(D):
        field[..., d] = np.real(np.fft.ifftn(anti_rho_hat * grad_hat[d]))
        
    return field


def _build_field(target, shape, sigma2_kernel, D, ftype):
    if ftype == "field":
        return target
    if ftype == "density":
        return _field_from_density(target, shape, sigma2_kernel, D)
    rho = _density_from_points(target, shape)
    return _field_from_density(rho, shape, sigma2_kernel, D)

def make_multi_scales_field_fun(target, ftype="points"):
    """ftype selects what `target` is:
      - "points":  point cloud, array (N, D), values in [0, 1)
      - "density": array of shape (512, 512) or (64, 64, 64)
      - "field":   array of shape (512, 512, 2) or (64, 64, 64, 3)
    See im2fields.py to turn an image into a density or field.
    """
    target = np.asarray(target)
    D = target.shape[1] if ftype == "points" else target.shape[-1] if ftype == "field" else target.ndim
    shape = (512, 512) if D == 2 else (64, 64, 64)
    cache = {}

    def get_field_fn(sigma2_kernel):
        key = (tuple(shape), float(sigma2_kernel))
        if key not in cache:
            cache[key] = jnp.asarray(
                _build_field(target, shape, sigma2_kernel, D, ftype)
            )
        field = cache[key]
        def spatial_grad(x):
            p = x.reshape(-1, D)
            idx = tuple(
                jnp.floor(p[:, d] * shape[d]).astype(jnp.int32) % shape[d]
                for d in range(D)
            )
            return clean_grad(field[idx].reshape(x.shape))
        return spatial_grad

    return get_field_fn

def make_target(target, sigma2, D):
    if target is None:
        def target_grad(x):
            return 0.0
        return target_grad
    if D not in (2, 3):
        raise ValueError(
            f"target is only supported for D in (2, 3), got D={D}"
        )
    target_ftype = "points"
    if isinstance(target, str):
        target = im2field(target)
        target_ftype = "density"
    get_field_fn = make_multi_scales_field_fun(target, ftype=target_ftype)
    target_grad = get_field_fn(sigma2)
    return target_grad