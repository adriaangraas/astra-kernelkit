API Summary
===========

Geometries
----------

.. autosummary::
   :toctree: _autosummary
      :maxdepth: 1
   kernelkit.VolumeGeometry
   kernelkit.ProjectionGeometry
   kernelkit.ProjectionGeometry.Beam
   kernelkit.resolve_volume_geometry
   kernelkit.experimental.suggest_volume_extent
   kernelkit.Detector
   kernelkit.GeometrySequence
   kernelkit.GeometrySequence.DetectorSequence
   kernelkit.rotate
   kernelkit.rotate_
   kernelkit.scale
   kernelkit.scale_
   kernelkit.shift
   kernelkit.shift_


Kernels
-------

.. autosummary::
   :toctree: _autosummary
      :maxdepth: 1

   kernelkit.kernel.BaseKernel
   kernelkit.kernel.copy_to_texture
   kernelkit.kernel.copy_to_symbol
   kernelkit.kernels.VoxelDrivenConeBP
   kernelkit.kernels.VoxelDrivenConeBP.TextureFetching
   kernelkit.kernels.RayDrivenConeFP


Projectors
----------

.. autosummary::
   :toctree: _autosummary
      :maxdepth: 1

   kernelkit.BaseProjector
   kernelkit.ConeBackprojector
   kernelkit.ConeProjector
   kernelkit.toolbox_support.ConeProjectorAdapter
   kernelkit.toolbox_support.ConeBackprojectorAdapter


Algorithms
----------

Reference algorithms:

.. autosummary::
   :toctree: _autosummary
      :maxdepth: 1

   kernelkit.bp
   kernelkit.fp
   kernelkit.fdk
   kernelkit.sirt

Operator-style algorithm building blocks: 

.. autosummary::
   :toctree: _autosummary
      :maxdepth: 1

   kernelkit.BaseOperator
   kernelkit.ProjectorOperator
   kernelkit.XrayTransform


Miscellaneous
-------------

Helpers for cone and Fourier-based algorithms:

.. autosummary::
   :toctree: _autosummary
      :maxdepth: 1

   kernelkit.processing.filter
   kernelkit.processing.preweight

Helpers for pitched 2D memory:

.. autosummary::
   :toctree: _autosummary
      :maxdepth: 1

   kernelkit.data.ispitched
   kernelkit.data.aspitched

Helpers for PyTorch:

.. autosummary::
   :toctree: _autosummary
      :maxdepth: 1

   kernelkit.torch_support.AutogradOperator
