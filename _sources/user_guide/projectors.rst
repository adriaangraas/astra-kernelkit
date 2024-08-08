.. _projectors:

==========
Projectors
==========

Projectors implement a tomographic **forward projection** or **backprojection**. In KernelKit, the projector objects manage the execution of a kernel and handle the geometries and data.

Overview
========

Currently added are the 3D forward projector and backprojectors.

+-------------------------------------+---------------------------------------------------------------------------------+
| Forward projectors                  |                                                                                 |
+=====================================+=================================================================================+
| :class:`kernelkit.ForwardProjector` | A projector for a list-based scan geometry and single 3D reconstruction volume. |
+-------------------------------------+---------------------------------------------------------------------------------+

+-----------------------------------+-----------------------------------------------------------------------------------+
| Backprojectors                    |                                                                                   |
+===================================+===================================================================================+
| :class:`kernelkit.BackProjector`  | A backprojector for a list-based scan geometry and single 3D volume.              |
+-----------------------------------+-----------------------------------------------------------------------------------+

Usage
=====

Projectors follow a simple interface, with getters, setters and deleters backing :python:`volume_geometry`, :python:`projection_geometry`, :python:`volume` and :python:`projections`. This is easiest to explain with an example:

.. code-block::

    # 1. Creating a forward projector.
    #    Passed arguments are used for compiling an optimal kernel.
    #    Here, the input is set to (z, y, x) axis order.
    fp = kk.ForwardProjector(volume_axes=(2, 1, 0))

    # 2. Setting attributes
    fp.volume_geometry     = vg
    fp.projection_geometry = pg
    fp.volume              = cp.asarray(x, dtype=cp.float32)
    fp.projections         = cp.empty((3, det.rows, det.cols), dtype=cp.float32)

    # 3. Calling the object executes the projector
    #    and stores the output in `projections`.
    fp()

    # 4. Attribute getters allow you to retrieve the object
    y = fp.projections


A full example is found at :github:`example/tutorials/projectors.ipynb`.