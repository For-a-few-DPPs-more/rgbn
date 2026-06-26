# blue-sampler

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/For-a-few-DPPs-more/rgbn/blob/main/examples.ipynb)

Generate large **stealthy point patterns** on the unit torus [0, 1)^D.

Stealthy point patterns have vanishing density fluctuations at low ("blue")
frequencies, making them useful for Monte Carlo integration, image
stippling, and any application that needs well-spread, low-discrepancy
points.

The main blue noise sampler (RGBN) implemented here have **linear** complexity in the 
number of points and the dimension, and run in under 15 minutes for 1 million 2D points.

## Links

- 🌐 [project website](https://for-a-few-dpps-more.github.io/rgbn/)
- 📦 [PyPI](https://pypi.org/project/blue_sampler/)
- 🐙 [GitHub repository](https://github.com/For-a-few-DPPs-more/rgbn)
- 📚 [documentation (Read the Docs)](https://blue_sampler.readthedocs.io)

## Installation

```bash
pip install blue_sampler
```

## Quick start

```python
import blue_sampler as blue

# 10 000 points in 2-D
x = blue.sample_points(N=10_000, D=2)
blue.plot(x, auto_zoom = True)
blue.plot_structure_factor(x)

# arbitrary dimension D
x = blue.sample_points(N=2_000, D=5)

# image stippling
x = blue.im2points(image="zebra.jpg")
```
![Zebra points](https://raw.githubusercontent.com/For-a-few-DPPs-more/rgbn/main/zebrapoints.png)

## Supported dimensions

| D    | Notes                                 |
|------|----------------------------------------|
| 2    | Fast, recommended for exploration      |
| 3    | ~2x slower than 2-D                    |
| 4–5  | Requires more iterations               |
| ≥ 6  | Experimental, for small sample sizes   |

## License

MIT