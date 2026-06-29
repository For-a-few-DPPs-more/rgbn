"""
blue_sampler
============

Generate stealthy point patterns — low-discrepancy, spectrally isotropic
samples on the unit torus [0, 1)^D.

Quick start
-----------
>>> import blue_sampler as blue
>>> x = blue.sample(N=10_000, D=2)          # (10000, 2) array
>>> blue.plot(x)
>>> blue.plot_structure_factor(x)
"""
from .api import im2points, sample_points, sample_tessels, sample_clusters, tile, from_geometry
from .viz import plot, plot_structure_factor, plot_tessels, plot_clusters
from .math import structure_factor
from .datasets import generate_dataset

__all__ = [
    "im2points",
    "sample_points",
    "sample_tessels",
    "sample_clusters",
    "tile",
    "from_geometry",
    "structure_factor",
    "generate_dataset",
    "plot",
    "plot_structure_factor",
    "plot_tessels",
    "plot_clusters",
]
