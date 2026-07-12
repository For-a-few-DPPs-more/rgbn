"""
Visualisation helpers.

plot                   Scatter plot of a 2-D or 3-D point set.
plot_structure_factor  Log-log plot of the radial structure factor S(k).
plot_tessels           Display a STIT tessellation across recursion depths.
plot_clusters          Display a cluster partition across recursion depths.
plot_polygons          Display an arbitrary collection of polygons.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from .math import structure_factor
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
    figsize: tuple = (8, 8),
    **scatter_kw,
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Scatter plot of a 2-D or 3-D point set.

    Parameters
    ----------
    points : ndarray of shape (N, D) or (N, M, D)
        Point coordinates, with D in {2, 3}.
        Arrays of shape (N, K, D) are flattened to (N*K, D) before
        plotting. Dimensions beyond 3 are silently dropped (only
        the first 3 coordinates are displayed).
    auto_zoom : bool, default False
        If True, large point sets are zoomed in so that at most
        ``max_points`` points are rendered, keeping the figure readable.
    max_points : int, default 30_000
        Maximum number of points to render when ``auto_zoom`` is True.
    ax : matplotlib Axes, optional
        Existing axes to draw into. If None, a new figure is created.
        For 3-D data, the axes must have ``projection="3d"``.
    return_fig : bool, default False
        If True, return ``(fig, ax)`` instead of calling ``plt.show()``.
    figsize : tuple, default (8, 8)
        Figure size in inches, forwarded to ``plt.figure()``.
        Ignored when ``ax`` is provided.
    **scatter_kw
        Extra keyword arguments forwarded to ``ax.scatter``
        (e.g. ``color``, ``s``, ``alpha``).

    Returns
    -------
    (fig, ax) if return_fig is True, else None.
    """
    pts = np.asarray(points).reshape(-1, np.asarray(points).shape[-1])
    D = min(pts.shape[-1], 3)
    pts = pts[:, :D]

    scale_min = pts.min()
    scale_max = pts.max()
    scale_delta = scale_max - scale_min

    if auto_zoom and (len(pts) > max_points):
        offset = np.random.rand(2)[None]
        zoom = (max_points / len(pts)) ** (1.0 / D)
        dtpts = pts / scale_delta - scale_min
        pts = pts[((1-zoom)*offset <= dtpts <= zoom + (1-zoom)*offset).all(axis=1)]

    kw = dict(s=10_000 / len(pts), color="black")
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
    min_val: float = 1e-20,
    ax: plt.Axes | None = None,
    return_fig: bool = False,
    **plot_kw,
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Log-log plot of the radial structure factor S(k).

    S(k) is estimated via scattering intensity (squared modulus of the
    empirical Fourier transform) at a set of wave vectors sampled up to
    a radius proportional to ``N^(1/D)``.

    Parameters
    ----------
    points : ndarray of shape (N, D)
        Point coordinates in [0, 1)^D.
    resolution : int, default 2000
        Number of sampled wave vectors. Higher values give a smoother curve
        at increased computation time.
    smoothed : bool, default True
        If True, overlay a local log-log Gaussian average on top of the raw
        scatter. If False, only the raw values are shown.
    min_val : float, default 1e-20
        Floor value applied before taking logarithms, to avoid ``log(0)``
        overflow.
    ax : matplotlib Axes, optional
        Existing axes to draw into. If None, a new figure is created.
    return_fig : bool, default False
        If True, return ``(fig, ax)`` instead of calling ``plt.show()``.
    **plot_kw
        Extra keyword arguments forwarded to ``ax.loglog`` (smoothed line).

    Returns
    -------
    (fig, ax) if return_fig is True, else None.

    Notes
    -----
    A stealthy / blue-noise point set will show S(k) ≈ 0 for small k.
    This is the visual signature of hyperuniformity.
    """
    pts = np.asarray(points).reshape(-1, np.asarray(points).shape[-1])
    k, S = structure_factor(pts, resolution=resolution)
    S = S.clip(min=min_val)

    if smoothed:
        logk = np.log(k)
        sigma = (logk[-1] - logk[0]) * 0.01
        logS = np.log(S)
        S_smooth = np.empty_like(logS)
        for i in range(len(k)):
            w = np.exp(-(logk - logk[i]) ** 2 / (2 * sigma**2))
            w /= w.sum()
            S_smooth[i] = np.exp(np.sum(w * logS))
        S = S.clip(min=S_smooth.min())
    else:
        S_smooth = np.nan

    kw = dict(marker="o", markersize=2, linewidth=2)
    kw.update(plot_kw)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    scat_color = "lightgray" if smoothed else "tab:blue"

    ax.set_axisbelow(True)
    ax.grid(True, which="both", alpha=0.4, zorder=0)
    bigS = S >= 30*S_smooth
    ax.scatter(k[~bigS], S[~bigS], s=5, color=scat_color, alpha=0.6, zorder=2)
    ax.scatter(k[bigS], S[bigS], s=20, color=scat_color, alpha=0.8, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if smoothed:
        ax.loglog(k, S_smooth, color="tab:blue", zorder=3, **kw)

    ax.set_xlabel(r"$k = \frac{2\pi}{L}\sqrt{n_x^2 + n_y^2\,\ldots}$")
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
    """Draw a set of quadrilaterals on ``ax`` (internal helper)."""
    for quad in tessels:
        qloop = np.vstack([quad, quad[0]])
        ax.plot(qloop[:, 0], qloop[:, 1], "-o", ms=3)
        ax.fill(qloop[:, 0], qloop[:, 1], alpha=0.25)
    return ax


def plot_tessels(
    tessels: NDArray,
    axes: plt.Axes | None = None,
    return_fig: bool = False,
) -> tuple[plt.Figure, NDArray] | None:
    """
    Display a STIT tessellation across successive recursion depths.

    The function starts from the provided tessellation and progressively
    merges pairs of quadrilaterals to reconstruct coarser levels, then
    plots each level side by side (up to 12 panels).

    Parameters
    ----------
    tessels : ndarray of shape (N, 4, 2)
        Output of `sample_tessels`. N must be a power of 2.
    axes : array of matplotlib Axes, optional
        Pre-existing axes to draw into. If None, a new figure is created
        with one subplot per recursion depth.
    return_fig : bool, default False
        If True, return ``(fig, axes)`` instead of calling ``plt.show()``.

    Returns
    -------
    (fig, axes) if return_fig is True, else None.

    Examples
    --------
    >>> ts = blue.sample_tessels(N=1024)
    >>> blue.plot_tessels(ts)
    """
    depth = int(np.log2(len(tessels)))
    n_plots = min(depth, 12)
    ncols = min(3, n_plots) if n_plots > 0 else 1
    nrows = int(np.ceil(n_plots / ncols))

    if axes is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).flatten()
    else:
        fig = axes.get_figure()

    while depth > 11:
        tessels = back_merge_tessels(tessels)
        depth -= 1

    for k in reversed(range(n_plots)):
        show_polygons(axes[k], np.random.permutation(tessels))
        axes[k].set_aspect("equal")
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
    """Scatter-plot a cluster partition on ``ax``, colour-coded by cluster (internal helper)."""
    n_clusters, n_points_per_cluster, _ = clusters.shape
    cmap = plt.get_cmap("tab20" if n_clusters > 10 else "tab10")

    flat_pts = clusters.reshape(-1, clusters.shape[-1])

    order = np.arange(len(flat_pts))
    if flat_pts.shape[-1] == 3:
        order = np.argsort(flat_pts[:, 2])

    cluster_indices = np.arange(n_clusters)[:, None]
    flat_indices = np.repeat(cluster_indices, n_points_per_cluster)

    colors = cmap(flat_indices % cmap.N)

    ax.scatter(flat_pts[order, 0], flat_pts[order, 1], s=4, color=colors[order])
    return ax


def plot_clusters(
    clusters: NDArray,
    axes: plt.Axes | None = None,
    return_fig: bool = False,
) -> tuple[plt.Figure, NDArray] | None:
    """
    Display a cluster partition across successive recursion depths.

    The function starts from the provided cluster set and progressively
    merges pairs of clusters to reconstruct coarser levels, then plots
    each level side by side (up to 9 panels), with each cluster shown
    in a distinct colour.

    Parameters
    ----------
    clusters : ndarray of shape (N, k, D)
        Output of `sample_clusters`. N must be a power of 2.
        D can be 2 or 3; higher dimensions are projected to the first 3
        coordinates via an isometric linear map before display.
    axes : array of matplotlib Axes, optional
        Pre-existing axes to draw into. If None, a new figure is created
        with one subplot per recursion depth.
    return_fig : bool, default False
        If True, return ``(fig, axes)`` instead of calling ``plt.show()``.

    Returns
    -------
    (fig, axes) if return_fig is True, else None.

    Examples
    --------
    >>> cl = blue.sample_clusters(N=512, D=2)
    >>> blue.plot_clusters(cl)
    """
    depth = int(np.log2(len(clusters)))
    n_plots = min(depth, 9)
    ncols = min(3, n_plots) if n_plots > 0 else 1
    nrows = int(np.ceil(n_plots / ncols))

    if axes is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_1d(axes).flatten()
    else:
        fig = axes.get_figure()

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
            [1,  1 / np.sqrt(3), -np.sqrt(2) / np.sqrt(3)],
            [-1, 1 / np.sqrt(3), -np.sqrt(2) / np.sqrt(3)],
            [0,  2 / np.sqrt(3),  np.sqrt(2) / np.sqrt(3)],
        ])
        clusters = np.einsum("ijk,kl->ijl", clusters, R)

    for k in reversed(range(n_plots)):
        show_clusters(axes[k], np.random.permutation(clusters))
        axes[k].set_aspect("equal")
        axes[k].axis("off")
        if k > 0:
            clusters = back_merge_clusters(clusters)

    plt.tight_layout()

    if return_fig:
        return fig, axes

    plt.show()
    return None


def plot_polygons(
    polygons: NDArray,
    ax: plt.Axes | None = None,
    return_fig: bool = False,
    color: str = "auto",
    linewidth: float = 1.2,
) -> tuple[plt.Figure, plt.Axes] | None:
    """
    Visualise a collection of polygons with colour-coded faces.

    Polygon opacity is scaled by inverse area squared, so tiny polygons
    (e.g. deep in a Pinwheel tiling) remain visible while large ones are
    rendered with full opacity.

    Parameters
    ----------
    polygons : ndarray of shape (N, M, 2)
        N polygons, each defined by M vertices in 2D.
        Typical inputs:

        - Output of `pinwheel_transform` — triangles of shape (N, 3, 2).
        - Output of `sample_tessels` — quadrilaterals of shape (N, 4, 2).
    ax : matplotlib Axes, optional
        Existing axes to draw into. If None, a new figure is created.
    return_fig : bool, default False
        If True, return ``(fig, ax)`` instead of calling ``plt.show()``.

    Returns
    -------
    (fig, ax) if return_fig is True, else None.

    Examples
    --------
    >>> pw = blue.pinwheel_transform(blue.pinwheel_base(), depth=4)
    >>> blue.plot_polygons(pw)
    """
    if len(polygons) >= 30_000:
        linewidth = 0
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    else:
        fig = ax.get_figure()

    n = len(polygons)
    if color == "auto":
        cmap = plt.cm.YlGnBu
        color = cmap(0.2 + 0.8 * (np.random.permutation(np.arange(n)) / n))

    x = polygons[:, :, 0]
    y = polygons[:, :, 1]
    x_next = np.roll(x, -1, axis=1)
    y_next = np.roll(y, -1, axis=1)
    areas_inv = 1 / np.abs(0.5 * np.sum(x * y_next - x_next * y, axis=1))
    areas_inv2 = areas_inv**2
    areas_inv2 = (areas_inv2 / areas_inv2.mean() + 0.01).clip(max=1.0)

    collection = PolyCollection(
        polygons,
        facecolors=color,
        edgecolors="none" if linewidth <= 0.01 else "#FFFFFF",
        linewidth=linewidth,
        alpha=areas_inv2,
    )
    ax.add_collection(collection)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()

    if return_fig:
        return fig, ax

    plt.show()