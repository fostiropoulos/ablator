# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../ablator'))

project = 'ablator'
copyright = '2023, Iordanis'
author = 'Iordanis'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'numpydoc',
              'sphinx.ext.todo', 'sphinx.ext.viewcode', 'nbsphinx',
              'sphinx.ext.autosummary']

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ['_static']

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_membersbool = False

html_title = "Ablator Documentation"
html_favicon = '_static/ablator-logo.svg'

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": [],
    "navbar_persistent": [],
    "header_links_before_dropdown": 3,
    "navigation_with_keys": False,
    "show_nav_level": 1,
    "home_page_in_toc": True,
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "repository_url": "https://github.com/fostiropoulos/ablator",
    "use_repository_button": True,
    "logo": {
        "image_light": "_static/ablator-banner-light.svg",
        "image_dark": "_static/ablator-banner-dark.svg",
        "link": "index",
        "alt_text": "ablator"
    },
    "search_bar_text": "Search"
}

html_sidebars = {
    "**": ["navbar-logo.html", "search-field.html", "sbt-sidebar-nav.html"]
}
