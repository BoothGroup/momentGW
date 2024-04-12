# docs/conf.py

import os
import sys

sys.path.insert(0, os.path.abspath("../"))

project = "momentGW"
copyright = "2024, Oliver J. Backhouse"
author = "Oliver J. Backhouse"

extensions = [
    "sphinx_mdinclude",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "numpydoc",
]

source_suffix = [".rst", ".md"]
master_doc = "index"

html_theme = "alabaster"
