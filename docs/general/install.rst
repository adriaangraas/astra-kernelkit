.. _installation:

Installation
============

Requirements
------------

 - An NVIDIA CUDA GPU with compute capability 3.0 or higher.
 - CuPy v11.0 or later.
 - Python 3.8 or later.

We highly recommend to install CuPy first, using the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html).


Python dependencies
^^^^^^^^^^^^^^^^^^^
This package relies on:
```
numpy
scipy
matplotlib
```
For detailed information on the dependencies, see the `setup.cfg` file.


Installation options
--------------------

Option 1: Installing KernelKit from PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Install `CuPy <https://docs.cupy.dev/en/stable/install.html>`_ first.


Option 2: Installing KernelKit from Conda-Forge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install `CuPy <https://docs.cupy.dev/en/stable/install.html>`_ first.
Using the Anaconda distribution, you can install ASTRA KernelKit from the
`conda-forge <https://conda-forge.org/>`_ channel:

.. code-block:: bash

    conda install -c conda-forge astra-kernelkit


Option 3: Installing KernelKit from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install `CuPy <https://docs.cupy.dev/en/stable/install.html>`_ first.
Installation from source is recommended if you want to contribute to
the development.

.. code-block:: bash

    git clone https://github.com/adriaangraas/astra-kernelkit.git
    cd astra-kernelkit
    pip install -e .
    pip install -e .[dev]  # for development
