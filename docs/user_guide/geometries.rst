.. _geometries:

==========
Geometries
==========

Geometries encode the positions of the source(s), detector(s), and object(s) for
multiple angles or timesteps. In KernelKit they are behave as plain-old Python
objects that hold values for the parameters required inside the projectors and
kernels. Different kernels may support different geometries (e.g., conebeam or parallel beam, 2D or 3D)
and may choose different parametrizations (e.g., support rotated detectors).

In KernelKit we introduce two definitions. A *scan geometry* encodes all
parameters associated with multiple projections. A *volume geometry* encodes the
position and rotation of the reconstruction volume at the center of the system.

.. _Overview:

Currently supported are a circular geometry and list-based geometry.
Additional geometries (2D/3D/4D) are welcome.

+--------------------------------------------+-------------------------------------------------------------------+
| Scan geometries                            |                                                                   |
+============================================+===================================================================+
| ``Sequence[kernelkit.ProjectionGeometry]`` | A :ref:`list-based flexible geometry<List-based scan geometries>` |
|                                            | for standard ASTRA kernels.                                       |
+--------------------------------------------+-------------------------------------------------------------------+

+------------------------------+--------------------------------------------------------+
| Volume geometries            |                                                        |
+==============================+========================================================+
| ``kernelkit.VolumeGeometry`` | Standard description of a                              |
|                              | :ref:`single reconstruction volume <Volume geometry>`. |
+------------------------------+--------------------------------------------------------+

.. _Conventions:

Conventions
===========

These conventions hold throughout for all Python code:

- **Right-handed coordinate system.**

  The :math:`z`-axis points along the axis of rotation (typically vertical), the
  :math:`x`-axis points (horizontal) to the right, and the :math:`y`-axis
  points (horizontal) into the page.

- **Vectors are in** :math:`(x, y, z)` **order.**

  Unless indicated otherwise, vector notation follows :math:`(x, y, z)`. An
  example are :python:`extent_min` and :python:`extent_max` arguments, that
  define the size of a reconstruction volume in a ``VolumeGeometry``.

  Note: data arrays, e.g., :cupy:`cupy.ndarray`, can still use alternative axis orders, such as :math:`(z, y, x)` in ASTRA Toolbox.

- **Angles are in radians, and use roll-pitch-yaw (RPY) format.**

  Roll
  is rotation about the :math:`x`-axis, pitch is rotation about the :math:`y`-axis,
  and yaw is rotation about the :math:`z`-axis. We take extrinsic Euler angles, meaning that the
  rotation is applied to the fixed coordinate system.

.. _Scan geometries:

.. _List-based scan geometries:

List-based scan geometries
==========================

ASTRA KernelKit allows working with arbitrary (non-circular) geometries.
Standard projectors require a Python :python:`Sequence[ProjectionGeometry]`, i.e., a list
or tuple with one geometry for each projection angle or timestep. List-based scan geometries 
can be viewed as object-oriented wrappers for ASTRA Toolbox vector geometries.
 
Single projection geometry
--------------------------

A :class:`kernelkit.ProjectionGeometry` is a description that parametrizes the
acquisition of a single-angle single-source single-detector set-up:

- a source position, :math:`\mathbf s \in \mathbb{R}^3`;
- a detector center, :math:`\mathbf d \in \mathbb{R}^3`;
- unit vectors :math:`\hat{\mathbf u} \in \mathbb{R}^3`, :math:`\hat{\mathbf v} \in \mathbb{R}^3`, the horizontal and vertical axes of the detector;
- a reference to a :class:`kernelkit.Detector` object.

.. code-block:: python

    from kernelkit import ProjectionGeometry, Detector, Beam

    # Create a detector
    detector = Detector(100, 100, pixel_width=1., pixel_height=1.)

    # Create a projection geometry
    geometry = ProjectionGeometry(
        source_position=[-100, 0, 0],  # Note: (x, y, z)
        detector_position=[100, 0, 0], # source-det are aligned on the x-axis
        u=[0, 1, 0],                   # pointing in the y-direction
        v=[0, 0, 1],                   # pointing in the z-direction
        detector=detector,
        beam=Beam.CONE,
    )



Remarks:

- Detector pixels are counted in the :math:`(-\hat{\mathbf u}, -\hat{\mathbf v})` direction. E.g., if :math:`z` points up, and :math:`\hat{\mathbf v} = (0, 0, 1)`, then the first row is the top row of the detector.
- :python:`Beam.CONE` and :python:`Beam.PARALLEL` denote divergent and parallel sources. For parallel beam, `source_direction` may be specified in place of `source_position`.

Building a list-based scan geometry
-----------------------------------

A list-based scan geometry can be assembled from a sequence of individual
``ProjectionGeometry`` objects. Alternatively, :func:`kernelkit.rotate` and
:func:`kernelkit.shift` to rotate or shift existing
geometries.

.. code-block:: python

    # 100 equidistantially spaced angles between 0 and 2π
    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)

    # Create 100 rotated copies around the z-axis, for a circular scan
    circular_geometry = [kernelkit.rotate(geometry, yaw=phi) for phi in angles]


Additional technical remarks:

- Functions :func:`kernelkit.rotate_` and :func:`kernelkit.shift_` exist to modify geometries in-place.
- List-based geometries are an array-of-structures type of object. :func:`kernelkit.ProjectionGeometrySequence` can be used to convert lists into an structure-of-arrays object that is more suitable for vectorized operations.
- Currently kernels do not support lists with mixed detectors.

.. _Volume geometry:

Volume geometry
===============

The volume geometry, :class:`kernelkit.VolumeGeometry`, is a data container for
the position, size, and rotation of the reconstruction volume. In CT it is
common to use a uniform discretization of the volume. The specification
therefore also requires a voxel size.

.. code-block:: python

    from kernelkit import VolumeGeometry, resolve_volume_geometry

    # Create a volume geometry
    cube = VolumeGeometry(
        shape=(100, 100, 100),  # 100x100x100 voxels
        extent_min=(-.5, -.5, -.5),  # lower corner
        extent_max=(.5, .5, .5),  # upper corner
        voxel_size=(.01, .01, .01),  # must use same units as ProjectionGeometry
    )

Alternatively, it can be easier to have some of the volume parameters to be
inferred automatically. This can be done by specifying :code:`None` for any
unknowns in the function :func:`kernelkit.resolve_volume_geometry`.

.. code-block:: python

    cube: VolumeGeometry = resolve_volume_geometry(
        shape=(None, None, None),
        extent_min=(-.5, -.5, -.5),
        extent_max=(.5, .5, .5),
        voxel_size=(.01, .01, .01),
    ) 

    print(cube.shape)  # [100, 100, 100]
