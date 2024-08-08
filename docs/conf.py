# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys
p = pathlib.Path(__file__).parents[1].resolve()
sys.path.insert(0, p.as_posix())

from kernelkit import toolbox_support
from kernelkit import torch_support

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ASTRA KernelKit'
copyright = '2023, ASTRA Toolbox & ASTRA Toolbox contributors'
author = 'Adriaan Graas'

# The full version, including alpha/beta/rc tags
release = "1.0.0-alpha.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "jupyter_sphinx",
    'numpydoc',
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinxcontrib.bibtex',
]

bibtex_bibfiles = ['refs.bib']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

extlinks = {
    'github': ('https://github.com/adriaangraas/astra-kernelkit/blob/main/%s', "%s"),
    'cupy': ('https://docs.cupy.dev/en/stable/reference/generated/%s.html', '%s'),
}

rst_prolog = """
.. role:: python(code)
    :language: python
    :class: highlight
"""

nbsphinx_execute = 'always'  # Always execute notebooks

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_title = 'ASTRA KernelKit'
# html_logo = "_static/astra_toolbox.png"
html_static_path = ['_static']
html_theme = 'sphinx_book_theme'
html_theme_options = {
    "repository_url": "https://github.com/adriaangraas/astra-kernelkit",
    "use_repository_button": True,
    "use_issues_button": False,
    "use_download_button": False,
    "use_fullscreen_button": False,
    "collapse_navbar": False,
    # "navbar_end": ["mybutton.html"],
    "announcement": "Pre-release version."
}
html_css_files = ["custom.css"]

autosummary_generate = True

autodoc_mock_imports = ['torch', 'torch.autograd']


def setup(app):
    app.config.html_context["default_mode"] = "light"
