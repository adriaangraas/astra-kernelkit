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

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ASTRA KernelKit'
copyright = '2023, ASTRA Toolbox & ASTRA Toolbox contributors'
author = 'Adriaan Graas'

# The full version, including alpha/beta/rc tags
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['numpydoc',
              'myst_parser',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.doctest']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_title = 'ASTRA KernelKit'
html_logo = "_static/astra_toolbox.png"
html_static_path = ['_static']
html_theme_options = {
    "repository_url": "https://github.com/adriaangraas/astra-kernelkit",
    "use_repository_button": True,
}

autosummary_generate = True

autodoc_mock_imports = ['torch', 'torch.autograd']

