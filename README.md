[![PyPI](https://img.shields.io/pypi/v/blue-sampler.svg?label=pypi&color=brightgreen)](https://pypi.org/project/blue-sampler/)
[![Docs](https://readthedocs.org/projects/blue-sampler/badge/?version=latest)](https://blue-sampler.readthedocs.io)
[![GitHub](https://img.shields.io/badge/source-GitHub-black?logo=github)](https://github.com/For-a-few-DPPs-more/rgbn)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/For-a-few-DPPs-more/rgbn/blob/main/examples.ipynb
)

# blue-sampler

Generate large **stealthy point patterns** on the unit torus [0, 1)^D. Stealthy point patterns have vanishing density fluctuations at low ("blue") frequencies, making them useful for Monte Carlo integration, image stippling, and any application that needs well-spread, low-discrepancy points. The main blue noise sampler (RGBN) implemented here have **linear** complexity in the number of points and the dimension, and run in under 15 minutes for 1 million 2D points.

## 📦 Installation

```bash
pip install blue_sampler
````

## 🚀 Quick start

```python
import blue_sampler as blue

# 10 000 points in 2D
x = blue.sample_points(N=10_000, D=2)
blue.plot(x, auto_zoom=True)

# structure factor visualization
blue.plot_structure_factor(x)

# higher dimensions
x = blue.sample_points(N=2_000, D=5)

# image stippling
x = blue.im2points(image="zebra.jpg")
```

---

## 🖼️ Example

![Zebra points](https://raw.githubusercontent.com/For-a-few-DPPs-more/rgbn/main/zebrapoints.png)

---

## 📊 Supported dimensions

| Dimension | Notes                              |
| --------- | ---------------------------------- |
| 2D        | Fast, recommended                  |
| 3D        | ~2× slower                         |
| 4–5D      | Works, more iterations needed      |
| ≥6D       | Experimental (small N recommended) |

---


## 📚 Links

* 🌐 Project website: [https://for-a-few-dpps-more/rgbn.github.io](https://for-a-few-dpps-more/rgbn.github.io)
* 📦 PyPI: [https://pypi.org/project/blue-sampler/](https://pypi.org/project/blue-sampler/)
* 🐙 GitHub: [https://github.com/For-a-few-DPPs-more/rgbn](https://github.com/For-a-few-DPPs-more/rgbn)
* 📖 Documentation: [https://blue-sampler.readthedocs.io](https://blue-sampler.readthedocs.io)

---

## 📄 License

MIT
