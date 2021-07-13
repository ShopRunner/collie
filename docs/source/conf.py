# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

from pathlib import Path
import sys

base_dir_loc = Path(__file__).parents[2]
version_loc = base_dir_loc / 'collie/_version.py'
with open(version_loc) as version_file:
    exec(version_file.read())

sys.path.insert(0, str(base_dir_loc))

# -- Project information -----------------------------------------------------

project = 'collie'
author = 'ShopRunner Data Science Team'
copyright = '2021, ShopRunner Data Science Team'

# The short X.Y version
version = __version__
# The full version, including alpha/beta/rc tags
release = __version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['m2r2',
              'sphinx_copybutton',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosectionlabel',
              'sphinx.ext.coverage',
              'sphinx.ext.napoleon',
              'sphinx.ext.viewcode',]
napoleon_custom_sections = [('Side Effects')]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

source_suffix = ['.rst', '.md']

master_doc = 'index'

language = 'en'

pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'  # need to install this pip intall sphinx-rtd-theme==0.4.3

html_theme_options = {
    'style_nav_header_background': '#4F6D5A',
}

html_show_sourcelink = True

html_logo = str(base_dir_loc / 'docs/images/collie-sticker.png')
html_favicon = str(base_dir_loc / 'docs/images/collie-favicon.ico')

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.css',
]
