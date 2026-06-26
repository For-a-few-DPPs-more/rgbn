"""
Visualisation helpers:
plot                  display a 2-D or 3-D point set
plot_structure_factor display the structure factor estimated through
                       scattering intensity
plot_tessels          display a tessellation at up to 12 recursion depths
plot_clusters         display a cluster partition at up to 12 recursion depths
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

from .math import structure_factor as _structure_factor
from .run.run_tessels import back_merge_tessels
from .run.run_clusters import back_merge_clusters


# ---------------------------------------------------------------------
# PLOT POINT SET
# ---------------------------------------------------------------------

def plot(
    points: NDArray,
    auto_zoom: bool = False,
    max_points: int = 30_000,
    ax: plt.Axes | None = None,
    return_fig: bool = False,
    figsize = (8, 8),
    **scatter_kw,
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Scatter plot of a 2-D or 3-D point set.

    If auto_zoom is set to True:
        For large point sets the view is automatically zoomed
        so that at most *max_points* points are displayed.

    Parameters
    ----------
    points : ndarray of shape (N, D)
        Point coordinates, with D in {2, 3}.
        Higher-dimensional arrays are silently projected onto the first 3 axes.
    auto_zoom : bool, default True
        Whether to apply auto_zoom for large point sets.
    max_points : int, default 30_000
        Maximum number of points to draw used for auto_zoom.
    ax : matplotlib Axes, optional
        Existing axes to draw into. If None, a new figure is created.
    return_fig : bool, default False
        If True, returns (fig, ax).
        If False, displays the figure and returns None.
    **scatter_kw
        Extra keyword arguments forwarded to `ax.scatter`.

    Returns
    -------
    (fig, ax) or None
    """
    pts = np.asarray(points).reshape(-1, np.asarray(points).shape[-1])
    D = min(pts.shape[-1], 3)
    pts = pts[:, :D]

    if auto_zoom and (len(pts) > max_points):
        zoom = (max_points / len(pts)) ** (1.0 / D)
        pts = pts[(pts <= zoom).all(axis=1)]

    kw = dict(s=10_000/len(pts), color="black")
    kw.update(scatter_kw)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        if D == 2:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    if D == 2:
        ax.scatter(pts[:, 0], pts[:, 1], **kw)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], **kw)

    ax.set_axis_off()
    plt.tight_layout()

    if return_fig:
        return fig, ax

    plt.show()
    return None


# ---------------------------------------------------------------------
# STRUCTURE FACTOR
# ---------------------------------------------------------------------

def plot_structure_factor(
    points: NDArray,
    resolution: int = 2000,
    smoothed: bool = True,
    ax: plt.Axes | None = None,
    return_fig: bool = False,
    **plot_kw,
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Log-log plot of the radial structure factor S(k).

    S(k) is estimated using standard scattering intensity.

    Parameters
    ----------
    points : ndarray of shape (N, D)
        Point coordinates in [0, 1)^D.
    resolution : int, default 2000
        Number of sampled wave vectors used to estimate sf.
    smoothed : bool, default True
        If True, apply a local log-log average. If False, plot raw values.
    ax : matplotlib Axes, optional
        Existing axes to draw into. If None, a new figure is created.
    return_fig : bool, default False
        If True, returns (fig, ax).
    **plot_kw
        Extra keyword arguments forwarded to `ax.loglog`.

    Returns
    -------
    (fig, ax) or None
    """
    pts = np.asarray(points).reshape(-1, np.asarray(points).shape[-1])
    k, S = _structure_factor(pts, resolution=resolution)

    if smoothed:
        logk = np.log(k)
        sigma = (logk[-1] - logk[0]) * 0.01
        logS = np.log(S)

        S_smooth = np.empty_like(logS)

        for i in range(len(k)):
            w = np.exp(-(logk - logk[i])**2 / (2 * sigma**2))
            w /= w.sum()
            S_smooth[i] = np.exp(np.sum(w * logS))
        
        S = S.clip(min = S_smooth.min()) 

    kw = dict(marker="o", markersize=2, linewidth=2)
    kw.update(plot_kw)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    scat_color = "lightgray" if smoothed else "tab:blue"

    ax.set_axisbelow(True)  # grille + ticks in background
    ax.grid(True, which="both", alpha=0.4, zorder=0)

    ax.scatter(k, S, s=5, color=scat_color, alpha=0.6, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if smoothed:
        ax.loglog(k, S_smooth, color="tab:blue", zorder=3, **kw)

    ax.set_xlabel(r"$k = \frac{2\pi}{L}\sqrt{n_x^2 + n_y^2…}$")
    ax.set_ylabel(r"$S(k)$")
    ax.set_title("Structure factor (log-log, scattering intensity)")
    plt.tight_layout()

    if return_fig:
        return fig, ax

    plt.show()
    return None


# ---------------------------------------------------------------------
# TESSELS
# ---------------------------------------------------------------------

def show_polygons(ax: plt.Axes, tessels: NDArray) -> plt.Axes:
    for quad in tessels:
        qloop = np.vstack([quad, quad[0]])
        ax.plot(qloop[:, 0], qloop[:, 1], '-o', ms=3)
        ax.fill(qloop[:, 0], qloop[:, 1], alpha=0.25)
    return ax


def plot_tessels(
    tessels: NDArray,
    return_fig: bool = False,
) -> tuple[plt.Figure, NDArray] | None:
    """
    Display a tessellation across up to 12 recursion steps.
    """
    depth = int(np.log2(len(tessels)))
    n_plots = min(depth, 12)
    ncols = min(3, n_plots) if n_plots > 0 else 1
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    while depth > 11:
        tessels = back_merge_tessels(tessels)
        depth -= 1

    for k in reversed(range(n_plots)):
        show_polygons(axes[k], np.random.permutation(tessels))
        axes[k].set_aspect('equal')
        axes[k].axis("off")

        if k > 0:
            tessels = back_merge_tessels(tessels)

    plt.tight_layout()

    if return_fig:
        return fig, axes

    plt.show()
    return None


# ---------------------------------------------------------------------
# CLUSTERS
# ---------------------------------------------------------------------

def show_clusters(ax: plt.Axes, clusters: NDArray) -> plt.Axes:
    n_clusters, n_points_per_cluster, _ = clusters.shape
    cmap = plt.get_cmap("tab20" if n_clusters > 10 else "tab10")

    flat_pts = clusters.reshape(-1, clusters.shape[-1])

    order = np.arange(len(flat_pts))
    if flat_pts.shape[-1] == 3:
        order = np.argsort(flat_pts[:, 2])

    cluster_indices = np.arange(n_clusters)[:, None]
    flat_indices = np.repeat(cluster_indices, n_points_per_cluster)

    colors = cmap(flat_indices % cmap.N)

    ax.scatter(flat_pts[order, 0], flat_pts[order, 1],
               s=4, color=colors[order])

    return ax


def plot_clusters(
    clusters: NDArray,
    return_fig: bool = False,
) -> tuple[plt.Figure, NDArray] | None:
    """
    Display a cluster partition across up to 12 recursion steps.
    """
    depth = int(np.log2(len(clusters)))
    n_plots = min(depth, 9)
    ncols = min(3, n_plots) if n_plots > 0 else 1
    nrows = int(np.ceil(n_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    while depth > 8:
        clusters = back_merge_clusters(clusters)
        depth -= 1

    clusters = np.random.permutation(
        clusters.transpose(1, 0, 2)
    ).transpose(1, 0, 2)[:, :16]

    D = clusters.shape[-1]
    if D > 3:
        clusters = clusters[:, :, :3]

    if D >= 3:
        R = np.array([
            [1, 1 / np.sqrt(3), -np.sqrt(2) / np.sqrt(3)],
            [-1, 1 / np.sqrt(3), -np.sqrt(2) / np.sqrt(3)],
            [0, 2 / np.sqrt(3), np.sqrt(2) / np.sqrt(3)],
        ])
        clusters = np.einsum("ijk,kl->ijl", clusters, R)

    for k in reversed(range(n_plots)):
        show_clusters(axes[k], np.random.permutation(clusters))
        axes[k].set_aspect('equal')
        axes[k].axis("off")

        if k > 0:
            clusters = back_merge_clusters(clusters)

    plt.tight_layout()

    if return_fig:
        return fig, axes

    plt.show()
    return None