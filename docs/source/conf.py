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

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

html_css_files = [
    'css/custom.css',
]

html_context = {
    "default_mode": "light"
}

numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_membersbool = False

html_title = "Ablator Documentation"
html_favicon = '_static/ablator-logo.svg'

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 6,
    "navigation_with_keys": False,
    "collapse_navigation": True,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "logo": {
        "image_light": "_static/ablator-banner-light.svg",
        "image_dark": "_static/ablator-banner-dark.svg",
        "link": "index",
        "alt_text": "ablator"
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/fostiropoulos/ablator",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    "icon_links_label": "Quick Links",

}

html_sidebars = {
    "**": ['search-field', 'sidebar-nav-bs'],
    "index": [],
    "notebooks/GettingStarted": [],
    "notebooks/GettingStarted-more-demos": [],
    "api.reference": [],
}
