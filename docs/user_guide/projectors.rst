.. _projectors:

==========
Projectors
==========

Projectors implement a tomographic **forward projection** or **backprojection**. In KernelKit, the projector objects manage the execution of a kernel and handle the geometries and data.

Overview
========

Currently added are the 3D forward projector and backprojectors.

+--------------------------------+---------------------------------------------------------------------------------+
| Forward projectors             |                                                                                 |
+================================+=================================================================================+
| ``kernelkit.ForwardProjector`` | A projector for a list-based scan geometry and single 3D reconstruction volume. |
+--------------------------------+---------------------------------------------------------------------------------+

+------------------------------+-----------------------------------------------------------------------------------+
| Backprojectors               |                                                                                   |
+==============================+===================================================================================+
| ``kernelkit.BackProjector``  | A backprojector for a list-based scan geometry and single 3D volume.              |
+------------------------------+-----------------------------------------------------------------------------------+

Usage
=====

Projectors follow a simple interface, with getters, setters and deleters backing :python:`volume_geometry`, :python:`projection_geometry`, :python:`volume` and :python:`projections`. This is easiest to explain with an example:

.. code-block::

    fp = kk.ForwardProjector(volume_axes=(0, 1, 2))

    # 2. Setting attributes is backed by projector intelligence
    fp.volume_geometry     = vg
    fp.projection_geometry = [x_proj, y_proj, z_proj]
    fp.volume              = cp.asarray(x, dtype=cp.float32)
    fp.projections         = cp.empty((3, det.rows, det.cols), dtype=cp.float32)

    # 3. Calling the object executes the projector
    fp()

    # 4. Getters allow you to retrieve the object
    y = fp.projections

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(cp.asnumpy(y[0]))
    axs[1].imshow(cp.asnumpy(y[1]))
    axs[2].imshow(cp.asnumpy(y[2]))
    plt.show()

    # 5. Deleters will clear GPU memory if there are no further references
    # del fp.projections
    cp.get_default_memory_pool().free_all_blocks()


