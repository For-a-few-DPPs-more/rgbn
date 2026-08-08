[![PyPI](https://img.shields.io/pypi/v/blue-sampler.svg)](https://pypi.org/project/blue-sampler/)
[![Docs](https://readthedocs.org/projects/blue-sampler/badge/?version=latest)](https://blue-sampler.readthedocs.io)
[![GitHub](https://img.shields.io/badge/source-GitHub-black?logo=github)](https://github.com/For-a-few-DPPs-more/rgbn)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/For-a-few-DPPs-more/rgbn/blob/main/examples.ipynb)


# blue-sampler

**Generate large stealthy point patterns** on the unit torus $[0, 1)^D$. 📐

Stealthy point patterns exhibit vanishing density fluctuations at low frequencies, making them particularly suited for **Monte Carlo** integration 🎯, **image stippling**, and any application requiring well-distributed, low-discrepancy points. 

The main blue-noise samplers (**RGBN** and **NUFFT**) offer **linear** complexity in both the number of points and the dimension. ⚡  
They can generate e.g. 1 million 2D points in under 15 minutes on a standard CPU, and up to 30× faster on GPU. 

> **Note**: Most methods implemented here support adaptive sampling from a target distribution. This feature is still experimental beyond 2d distributions

> **Positioning and scope**: There are many alternative approaches for blue-noise and hyperuniform point sampling, with an abundant literature from both theoretical and practical perspectives. These methods target different trade-offs between spectral quality, anisotropy, computational cost, dimensionality, implementation complexity, hardware requirements, etc.

For example, some algorithms such as parallel Poisson-disk sampling are primarily designed for fast practical generation of visually convincing blue-noise patterns, where a moderate level of spectral suppression may be sufficient. Some algorithms are probably very interesting from a theoretical perspective, but reusable code can be hard to find. Other methods are highly optimized for specific domains such as 2D or 3D graphics. Some methods benefit strongly from GPU acceleration, like standard Gaussian blue noise, whereas others are designed to remain efficient on the CPU, like CCVT. Some rely on externally precomputed special sequences, like Sobol, but are thus very fast to use if the precomputed table is available. Notably, the FReSCo library provides very high-quality and scalable blue-noise samples based on non uniform fourier transform, but requires heavy spatial dependencies that need to be installed before using the software. All of these aspects are interesting and should be balanced according to the user's particular practical context.

"blue-sampler" is therefore not intended as a universal sampling method, but is primarily intended as a lightweight Python framework for mathematical and computational experiments on blue noise and hyperuniformity, where stronger spectral constraints (e.g. $S(k) \lesssim 10^{-3}$–$10^{-4}$ in the low-frequency regime), experiments beyond the usual 2D/3D settings, and rapid experimentation with different constructions are of interest.»
---

## 📦 Installation

```bash
pip install blue-sampler
```

---

## Quick Start

```python
import blue_sampler as blue

# Generate 10,000 2D blue-noise points
x = blue.sample_points(N=10_000, D=2)
blue.plot(x) 📈

# Structure factor
blue.plot_structure_factor(x) 📊

# Image stippling
x = blue.im2points("zebra.jpg") 🖼️ #return points
x = blue.im2quads("vangogh.jpg")   #quadrilaterals
```

---

## 🖼️ Example

<p align="center">
  <img src="https://raw.githubusercontent.com/For-a-few-DPPs-more/rgbn/main/zebrapoints.png" width="650" alt="Blue noise stippling example">
</p>

---

## 📋 Available Samplers

### Main sampling methods

```python
x = blue.sample_points(N, D, method="rgbn") #(N, D)
```

| Method         | Description                              |
|----------------|------------------------------------------|
| `rgbn`         | Recursive Gaussian-Blue Noise 🔄         |
| `nufft`        | Non-Uniform Fast Fourier Transform 📈    |
| `bruteforce`   | Base GBN sampler (best quality, slower)  |

---

## Alternative Samplers

### Sobol sequence

```python
x = blue.sobol(N, D) #(N, D)
```
Low-discrepancy quasi-random sequence. 📏

### Clusters

```python
# Raw clusters
cl = blue.sample_clusters(N, D) #(N, K, D)
blue.plot_clusters(cl) 🔗

# Convert to point set
x = blue.cluster2points(cl) #(N, m, D)
```

### STIT Tessellations (2D only)

```python
# Raw STIT tessellation (quadrilaterals)
ts = blue.sample_tessels(N) #(N, 4, D=2)
blue.plot_tessels(ts) 🧩

# Convert to point set
x = blue.tessel2points(ts) #(N, m, D=2)
```

### Pinwheel Tilings (2D only)

```python
# Base pinwheel triangle
pw0 = blue.pinwheel_base() 𖣘 #(3, D=2)

# Triangulation level 4
pw4 = blue.pinwheel_transform(pw0, depth=4) #(4*5**depth, 3, D=2)
blue.plot_polygons(pw4) 🔺

#===============================
# Convert Pinwheel to point set:
#===============================
#sample points from the BASE pinwheel
x0 = blue.tessel2points(pw0) #(m, D=2)
#then replicate the sample on the full triangulation,
#in a fractal way
x4 = blue.pinwheel_transform(x0, depth = 4) #(4*5**depth, m, D=2)
```

> **Note**: The conversion from geometric objects (polygons or clusters) to point sets is performed using a standard **moment matching** technique. cluster2points(x, p = 3) and tessel2points(x, p = 3) will sample m points per batch that mimic the statistical {0, 1, ... p-1} moments of the batch.

---

## Supported Dimensions

| Dimension | Status       |
|-----------|--------------|
| 2–3D      | Fast ⚡       |
| 4–5D      | Supported    |
| ≥6D       | Experimental 🧪 |

---

## Documentation & Links 🔗

- 📖 **Documentation**: [https://blue-sampler.readthedocs.io](https://blue-sampler.readthedocs.io)
- 📦 **PyPI**: [https://pypi.org/project/blue-sampler](https://pypi.org/project/blue-sampler/)
- 🌐 **Project website**: [https://for-a-few-dpps-more.github.io/rgbn/](https://for-a-few-dpps-more.github.io/rgbn/)
- 💻 **GitHub**: [https://github.com/For-a-few-DPPs-more/rgbn](https://github.com/For-a-few-DPPs-more/rgbn)

---

## 📚 References

The algorithms and mathematical tools implemented in **blue-sampler** are based or inspired from the following works.

- **Gaussian Blue Noise (repulsive interaction kernel)**  
  *A. G. M. Ahmed, J. Ren, and P. Wonka.*  
  **Gaussian Blue Noise.**  
  *ACM Transactions on Graphics (SIGGRAPH Asia), 41(6), 2022.*  
  DOI: 10.1145/3550454.3555519

- **FReSCo (Non uniform FFT)**  
  *A. Shih, M. Casiulis, and S. Martiniani.*  
  **Fast Generation of Spectrally-Shaped Disorder.**  
  *Physical Review E*, 110(3):034122, 2024.  
  DOI: 10.1103/PhysRevE.110.034122

- **STIT tessellations and moment matching**  
  *L. Lotz and M. A. Klatt.*  
  **Persistence of asymptotic variance under transport: from hyperfluctuation to stealthy hyperuniformity.**  
  *arXiv:2605.22803*, 2026.

- **Aperiodic tiling for hyuperuniformity (here pinwheel)**  
  *A. Gabrielli, B. Jancovici, M. Joyce, J. L. Lebowitz, L. Pietronero, and F. Sylos Labini.*  
  **Generation of Primordial Cosmological Perturbations from Statistical Mechanical Models.**  
  *Physical Review D*, **67**(4):043506, 2003.  

- **SquareNet** [(v1.3.11). 2026](https://squarenet.readthedocs.io/en/latest/).  
  Grid data structure for point clouds, codeveloped with RGBN \
  for efficient neighbor query and fourier trnasform in Python.

- **Sobol sequences**  
  Wrapped from `scipy.stats.qmc.Sobol` (SciPy).

---

## License

MIT
