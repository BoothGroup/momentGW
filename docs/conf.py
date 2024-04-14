# docs/conf.py

import os
import sys

sys.path.insert(0, os.path.abspath("../"))

project = "momentGW"
copyright = "2024, Booth Group"
author = "Booth Group"

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx_mdinclude",
    "sphinx_rtd_theme",
    "autoapi.extension",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]

autoapi_dirs = ["../momentGW"]
autoapi_options = [
    "members",
    "inherited-members",
    "show-inheritance",
]
autoapi_member_order = "bysource"
autoapi_add_toctree_entry = False
autoapi_python_use_implicit_namespaces = True
autoapi_own_page_level = "class"
autoapi_template_dir = "_templates/autoapi"

highlight_language = "python3"
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
source_suffix = [".rst", ".md"]
master_doc = "index"

html_theme = "sphinx_rtd_theme"
html_sidebars = {
    "**": [
        "about.html",
        "badges.html",
        "navigation.html",
        "relations.html",
        "searchbox.html",
    ]
}

def setup(app):
    """Setup function for Sphinx.
    """
    pass
