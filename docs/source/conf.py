# docs/source/conf.py

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

# -- Project information -----------------------------------------------------

project = "blue-sampler"
author = "Armand de Cacqueray"
copyright = "2026, Armand de Cacqueray"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

# Generate autosummary pages automatically
autosummary_generate = True

# Sensible defaults for API docs
autodoc_default_options = {
    "members": True,
    "imported-members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Keep type hints in the signature:
# sample_points(N: int, D: int = 2)
autodoc_typehints = "signature"

# Parse Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]