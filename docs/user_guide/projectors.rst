.. _user_guide:

Projectors
==========

Projectors implement a tomographic **forward projection** or **backprojection**, using a single volume and a single set of projections. They guarantee valid usage of their underlying projection kernels, and act efficiently on (re)projection, and in-place updates of data.  

Usage example
-------------


Interface
---------

 - Projectors must implement `kernelkit.BaseProjector` to serve as a backend in algorithms and operators.


.. code-block:: console

   (.venv) $ pip install lumache
