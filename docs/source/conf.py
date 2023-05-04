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

extensions = ['sphinx.ext.autodoc','numpydoc',
    'sphinx.ext.todo','sphinx.ext.viewcode']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_membersbool=False

# def maybe_skip_member(app, what, name, obj, skip, options):
#     print("app is ", app, "\n")
#     print("what is ", what, "\n")
#     print("name is ", name, "\n")
#     print("obj is ", obj, "\n")
#     print("skip is ", skip, "\n")
#     print("options is ", options, "\n\n\n")
#     if isinstance(obj, )

#     return None

# def setup(app):
#     app.connect('autodoc-skip-member', maybe_skip_member)