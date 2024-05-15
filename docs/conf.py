"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import time

from sphinx_pyproject import SphinxConfig

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# import data from pyproject.toml using https://github.com/sphinx-toolbox/sphinx-pyproject
# additional data can be added with `[tool.sphinx-pyproject]` and retrieved with `config['']`.
config = SphinxConfig("../pyproject.toml")  # add `, globalns=globals()` to directly insert in namespace
year = time.strftime("%Y")

project = "Autotuning Methodology"
author = config.author
copyright = f"2022-{year}, {author}"
version = config.version  # major version (e.g. 2.6)
release = config.version  # full version (e.g. 2.6rc1)


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.githubpages",
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
add_module_names = False
toc_object_entries_show_parents = "hide"

# -- Options for inheritance diagrams ----------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/inheritance.html
inheritance_graph_attrs = dict(rankdir="TB", fontsize=24, ratio="compress")
inheritance_node_attrs = dict(fontsize=24)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "source/logo_autotuning_methodology.svg"
html_theme_options = {
    "logo_only": True,
}
