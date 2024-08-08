=============================
ASTRA KernelKit documentation
=============================

**ASTRA KernelKit** is an all-Python tomographic reconstruction package,
leveraging the GPU-accelerated tomographic projection kernels of the ASTRA
Toolbox using CuPy.

- Develop, explore and debug CT algorithms and X-ray projectors with the Python ecosystem.
- Leverage CUDA features, such as CUDA graphs or unified memory, for maximum performance.
- Kernels are compiled during the script with NVRTC and
  `Jinja2 <https://jinja.palletsprojects.com/>`_. Hence, no
  complicated build system, and yet easy customization of kernels.
- Easy to install and integrates well with PyTorch and other GPU frameworks!

Try ASTRA KernelKit yourself, visit :doc:`general/install`.


.. note::
   For **ASTRA Toolbox** visit `<https://astra-toolbox.readthedocs.io/en/latest/>`_.


.. toctree::
   :maxdepth: 2
   :hidden:

   Installation <general/install>
   X-ray CT projectors <general/tomography>
   About / FAQ <general/about>


.. toctree::
   :caption: Reference
   :maxdepth: 2
   :hidden:

   Geometries <user_guide/geometries>
   Projectors <user_guide/projectors>
   reference/api_summary

.. toctree::
   :caption: Tutorials
   :maxdepth: 2
   :hidden:

   KernelKit concepts <tutorials/concepts>

