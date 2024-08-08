.. _user_guide:

Kernels
=======

Kernel objects are wrappers around CUDA source code and CuPy `cupy.RawModule`.
 - 

Usage Example
-------------
.. code-block:: python

   (.venv) $ pip install lumache


Interface
---------
Formal requirements are:
 - `Kernel` must not interfere with CUDA graphs, in particular it may not perform D2H transfers, or perform GPU allocations.
 - A `Kernel.__call__(*args, **kwargs)` must launch one or multiple kernels. It must not return a value.
 - `Kernel.compile()` (re)compiles the kernel.

A kernel class may subclass `kernelkit.Kernel` to inherit default compilation behavior.
