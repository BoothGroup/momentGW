# docs/conf.py

import os
import sys

from momentGW.base import Base

sys.path.insert(0, os.path.abspath("../"))

project = "momentGW"
copyright = "2024, Oliver J. Backhouse"
author = "Oliver J. Backhouse"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_mdinclude",
    "numpydoc",
]

templates_path = ["_templates"]

autodoc_default_options = {
    "show-inheritance": True,
    "members": True,
    "private-members": False,
    "special-members": True,
    "undoc-members": False,
    "member-order": "groupwise",
    "typehints": "description",
}

def autodoc_skip_member(app, what, name, obj, skip, options):
    """Rule to skip members.
    """
    return None

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = True
numpydoc_class_members_toctree = False

highlight_language = "python3"
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
source_suffix = [".rst", ".md"]
master_doc = "index"

html_theme = "alabaster"
html_theme_options = {
    "fixed_sidebar": True,
    "badge_branch": "master",
    "github_user": "BoothGroup",
    "github_repo": "momentGW",
    "github_button": False,
    "extra_nav_links": {
        "GitHub": "https://github.com/BoothGroup/momentGW",
        "Report Issues": "https://github.com/BoothGroup/momentGW/issues",
    },
}
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
    app.connect("autodoc-skip-member", autodoc_skip_member)
