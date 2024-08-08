.. _differences:

About / FAQ
===========

ASTRA KernelKit is a side-project of ASTRA Toolbox, and aims to provide
more flexible access to the tomographic projectors and kernels used in ASTRA.
It is a minimalistic standalone library.

How does KernelKit work?
^^^^^^^^^^^^^^^^^^^^^^^^

Tomographic projectors are often written in a low-level language, like
C/C++, and use NVIDIA CUDA for acceleration on graphical processing units (GPUs).

ASTRA KernelKit makes it easy to use Python to access and modify these
projectors. It does so by using CuPy, which in turn uses NVRTC, 
the NVIDIA runtime compiler, to compile CUDA kernels during the Python script.

.. figure:: idea.png
    :alt: A comparison of KernelKit with ASTRA Toolbox and Tomosipo.
    
    The comparison of KernelKit with ASTRA Toolbox and Tomosipo is explained
    in the paper publication associated with the software package :footcite:p:`graas_2024`.


ASTRA KernelKit or ASTRA Toolbox?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

KernelKit:

-  directly compile and call ASTRA Toolbox kernels in your Python code;
-  have more flexibility toward data axis order, data types, and
   memory layout, helping to get better performance;
-  customize ASTRA Toolbox kernels, for, e.g., machine learning applications.
-  integrate kernels with advanced CUDA, such as unified memory and streams.

Toolbox:

-  pre-implemented algorithms (e.g., SIRT, CGLS, FBP, etc.);
-  larger variety of kernels and functionalities;
-  multi-GPU support out of the box;
-  Windows and MatLab support.


Why CuPy?
^^^^^^^^^

A frequently asked question is why KernelKit is based on CuPy, and not
any other GPU array library such as PyTorch or PyCUDA.

A main feature of CuPy is that it transparently exposes advanced CUDA
functionality to Python, such as the NVIDIA run-time compiler, but also
streams, graphs, CUDA arrays, and texture memory. This allows KernelKit to
be written with a minimalistic code base and without shipping binary code.

Is KernelKit compatible with ...?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**ASTRA Toolbox**: KernelKit does not aim to be a drop-in replacement for ASTRA
Toolbox. There are ``ForwardProjector`` and ``BackProjector`` ASTRA Toolbox
compatible classes available, making it easy to cross-test
implementations with the ASTRA Toolbox.

**Tomosipo**: Work in progress in a `Github branch <https://github.com/adriaangraas/tomosipo/tree/kernelkit-integration>`_.
Here a KernelKit operator can be created with :python:`ts.operator(..., backend='kernelkit')`.

**PyTorch**: Torch tensors can be converted to CuPy arrays and vice versa
efficiently, without incurring copies. See :class:`kernelkit.torch_support.AutogradOperator`.

I found an issue / How can I contribute?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Contributions welcome! Please report any issues on the Github issue tracker.

.. footbibliography::