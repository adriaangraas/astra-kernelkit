.. _installation:

============
Installation
============


Requirements
============
ASTRA KernelKit is a Python package with requirements:

- An NVIDIA CUDA-enabled GPU with a compute capability 3.0 or higher.
- CuPy v12.0 or later.
- Python 3.10 or later.

This package furthermore relies on a few Python dependencies, such as ``numpy``. For detailed information on the dependencies, see the `setup.cfg` file.


1. Install `cupy` first
=======================
It is better to install CuPy yourself, rather than pulling it in as a dependency, since this leaves you with more flexibility towards the CUDA version and installation method. We recommend following `CuPy instructions <https://docs.cupy.dev/en/stable/install.html>`_. KernelKit does not require installation of additional packages (such as cuDNN), and is not dependent on a specific CUDA version.


2. Then install `astra-kernelkit`
=================================

Option 1: Using PyPI
--------------------

.. code-block:: bash

    pip install astra-kernelkit



Option 2: using Conda-Forge
---------------------------

If you are using an Anaconda environment, you can install ASTRA KernelKit from the
`conda-forge <https://conda-forge.org/>`_ channel:

.. code-block:: bash

    conda install -c conda-forge astra-kernelkit

There are no particular advantages to a conda package.


Option 3: from source
---------------------

Installation from source is recommended if you want to contribute to
the development.

.. code-block:: bash

    git clone https://github.com/adriaangraas/astra-kernelkit.git
    cd astra-kernelkit
    pip install -e .
    # alternatively: use "pip install -e .[dev]" for development