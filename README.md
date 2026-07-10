[![PyPI](https://img.shields.io/pypi/v/blue-sampler.svg)](https://pypi.org/project/blue-sampler/)
[![Docs](https://readthedocs.org/projects/blue-sampler/badge/?version=latest)](https://blue-sampler.readthedocs.io)
[![GitHub](https://img.shields.io/badge/source-GitHub-black?logo=github)](https://github.com/For-a-few-DPPs-more/rgbn)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/For-a-few-DPPs-more/rgbn/blob/main/examples.ipynb)

# blue-sampler

**Generate large stealthy point patterns** on the unit torus $[0, 1)^D$. 📐

Stealthy point patterns exhibit vanishing density fluctuations at low frequencies, making them particularly suited for **Monte Carlo integration** 🎯, **image stippling**, and any application requiring well-distributed, low-discrepancy points. 

The main blue-noise samplers (**RGBN** and **NUFFT**) offer **linear** complexity in both the number of points and the dimension. ⚡  
They can generate e.g. 1 million 2D points in under 15 minutes on a standard CPU, and up to 30× faster on GPU. 

> **Note**: Most implemented methods support adaptive sampling from a target distribution by passing an array of points (sampled i.i.d. from the desired distribution) to the `target` argument.

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
x = blue.im2points("zebra.jpg") 🖼️
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
x = blue.sample_points(N, D, method="rgbn")
```

| Method         | Description                              |
|----------------|------------------------------------------|
| `rgbn`         | Recursive Green-Blue Noise 🔄            |
| `nufft`        | Non-Uniform Fast Fourier Transform 📉    |
| `bruteforce`   | Base GBN sampler (best quality, slower)  |

---

## Alternative Samplers

### Sobol sequence

```python
x = blue.sobol(N, D)
```
Low-discrepancy quasi-random sequence. 📏

### STIT Tessellations

```python
# Raw STIT tessellation
ts = blue.sample_tessels(N, D)
blue.plot_tessels(ts) 🧩

# Convert to point set
x = blue.tessel2points(ts)
```

### Clusters

```python
# Raw clusters
cl = blue.sample_clusters(N, D)
blue.plot_clusters(cl) 🔗

# Convert to point set
x = blue.cluster2points(cl)
```

### Pinwheel Tilings

```python
# Base pinwheel triangle
pw0 = blue.pinwheel_base()

# Triangulation level 4
pw4 = blue.pinwheel_transform(pw0, depth=4)
blue.plot_polygons(pw4) 🔺

# Convert Pinwheel to point set
x = blue.pinwheel_transform(blue.tessel2points(pw0))
```

> **Note**: The conversion from geometric objects (polygons or clusters) to point sets is performed using a standard **moment matching** technique.

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

- **Gaussian Blue Noise**  
  *A. G. M. Ahmed, J. Ren, and P. Wonka.*  
  **Gaussian Blue Noise.**  
  *ACM Transactions on Graphics (SIGGRAPH Asia), 41(6), 2022.*  
  DOI: 10.1145/3550454.3555519

- **FReSCo (NUFFT sampler)**  
  *A. Shih, M. Casiulis, and S. Martiniani.*  
  **Fast Generation of Spectrally-Shaped Disorder.**  
  *Physical Review E*, 110(3):034122, 2024.  
  DOI: 10.1103/PhysRevE.110.034122

- **STIT tessellations and moment matching**  
  *L. Lotz and M. A. Klatt.*  
  **Persistence of asymptotic variance under transport: from hyperfluctuation to stealthy hyperuniformity.**  
  *arXiv:2605.22803*, 2026.

- **Pinwheel**  
  *S. Torquato and F. H. Stillinger.*  
  **Local density fluctuations, hyperuniformity, and order metrics.**  
  *Physical Review E*, 68(4):041113, 2003.

- **SquareNet**  
  *A. de Cacqueray.*  
  **SquareNet** (v0.1.0). 2026.

- **Sobol sequences**  
  Wrapped from `scipy.stats.qmc.Sobol` (SciPy).

---

## License

MIT