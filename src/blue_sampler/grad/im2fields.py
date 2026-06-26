"""
Turn an image into a density usable as `target` in fields.py
(make_multi_scales_field_fun(im2field(path), ftype="density"))
"""
from __future__ import annotations
import numpy as np

def im2field(path, shape=(512, 512), invert=True):
    """Load any image (color, grayscale, any size/ratio) and resample it
    into a normalized density on `shape`. Pixels close to white get high
    density unless `invert=True` (use that for dark strokes on white)."""
    from PIL import Image
    img = Image.open(path).convert("L").resize(shape[::-1], Image.LANCZOS)
    rho = np.asarray(img, dtype=np.float32) / 255.0
    rho = (rho.T)[:, ::-1]
    if invert:
        rho = 1.0 - rho
    return rho / rho.sum()