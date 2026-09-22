# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path Setup --------------------------------------------------------------
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../"))

# import scripts.main
import scripts.ms_cross_validation
import scripts.validate_geojson
import scripts.parse_args
import scripts.osm_api
import scripts.parse_geojson
import scripts.geometry_manipulation


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NLR GHE Screen'
copyright = '2026, Joseph Barrows'
author = 'Joseph Barrows'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
