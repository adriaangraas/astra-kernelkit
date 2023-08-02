from typing import Any, Callable, List, Optional, Tuple
import cupy as cp
import numpy as np
from abc import ABC

from astrapy.projector import BaseProjector
from astrapy.geom.proj import ProjectionGeometry
from astrapy.geom.vol import VolumeGeometry

import astra
import astra.experimental


class CompatProjector(BaseProjector, ABC):
    """Base class for ASTRA projectors."""

    __slots__ = ('_volume', '_projections', '_vol_geom', '_proj_geom')


    def __init__(self):
        """Base class for ASTRA projectors."""
        super().__init__()

    @property
    def volume(self) -> cp.ndarray:
        """The volume to project from."""
        return self._volume

    @volume.setter
    def volume(self, value: cp.ndarray):
        """The volume to project from.

        If the volume is of the same shape as before, the ASTRA projector
        will not be reinitialized.

        Parameters
        ----------
        value : array-like
            The volume to project from."""
        if not hasattr(self, 'volume_geometry'):
            expected_shape = tuple(reversed(self.volume_geometry.shape))
            if expected_shape != value.shape:
                raise ValueError(
                    f"Expected volume of shape {expected_shape} (z, y, x)"
                    f" according to the volume geometry, but got "
                    f"{value.shape} instead.")

        # Make sure CuPy doesn't clean up the volume while ASTRA is busy
        assert value.dtype == cp.float32
        assert value.flags.c_contiguous
        self._volume = value
        z, y, x = self._volume.shape
        # This operation is by my knowledge always efficient, so does not
        # require caching.
        self._astra_vol_link = astra.pythonutils.GPULink(
            self._volume.data.ptr, x, y, z, x * 4)

    @volume.deleter
    def volume(self):
        """Delete the volume and clean up ASTRA projector."""
        del self._volume
        del self._astra_vol_link  # TODO: does this delete?
        try:
            del self._astra_projector_id
        except AttributeError:
            pass

    @property
    def volume_axes(self):
        """The axes of the volume to project from."""
        return (2, 1, 0)

    @property
    def projections(self) -> cp.ndarray:
        """The projections onto which to project."""
        try:
            return self._projections
        except AttributeError:
            raise AttributeError("No projections have been set yet.")

    @projections.setter
    def projections(self, value: cp.ndarray):
        """The projections onto which to project."""
        assert value.dtype == cp.float32
        assert value.flags.c_contiguous

        if not hasattr(self, 'projection_geometry'):
            expected_shape = (self.projection_geometry[0].detector.rows,
                              len(self.projection_geometry),
                              self.projection_geometry[0].detector.cols)
            if value.shape != expected_shape:
                raise ValueError(f"Projection shape does not match projection"
                                 f" geometry. Got {value.shape}, "
                                 f" expected {expected_shape}.")

        assert value.dtype == cp.float32
        assert value.flags.c_contiguous
        z, y, x = value.shape
        self._projections = value
        self._astra_proj_link = astra.pythonutils.GPULink(
            self._projections.data.ptr, x, y, z, x * 4)

    @projections.deleter
    def projections(self):
        """Delete the projections and clean up ASTRA projector."""
        del self._projections
        del self._astra_proj_link  # TODO: does this delete?
        del self._astra_projector_id

    @property
    def projection_axes(self):
        """The axes of the volume to project from."""
        return (1, 0, 2)

    @property
    def volume_geometry(self) -> VolumeGeometry:
        return self._vol_geom

    @volume_geometry.setter
    def volume_geometry(self, value: VolumeGeometry):
        try:
            del self.volume_geometry  # invalidate projector
        except AttributeError:
            pass

        if hasattr(self, 'volume'):
            expected_shape_volume = tuple(reversed(self.volume.shape))
            if value.shape != expected_shape_volume:
                raise ValueError(
                    f"The given volume geometry does not match the "
                    f"shape of the volume in the projector. Got "
                    f"{value.shape} in (x, y, z) convention of the "
                    f"volume geometry, but the volume shape is "
                    f"{self.volume.shape} in (z, y, x) convention.")

        self._vol_geom = value
        vol_shp = value.shape
        ext_min = value.extent_min
        ext_max = value.extent_max
        self._astra_vol_geom = astra.create_vol_geom(
            vol_shp[1], vol_shp[0], vol_shp[2],
            ext_min[1], ext_max[1],
            ext_min[0], ext_max[0],
            ext_min[2], ext_max[2],
        )

    @volume_geometry.deleter
    def volume_geometry(self):
        del self._vol_geom
        del self._astra_vol_geom  # TODO: does this delete?
        try:
            del self._astra_projector_id
        except AttributeError:
            pass

    @property
    def projection_geometry(self):
        return self._proj_geom

    @projection_geometry.setter
    def projection_geometry(self, value: List[ProjectionGeometry]):
        try:
            del self.projection_geometry  # invalidate projector
        except AttributeError:
            pass

        if hasattr(self, 'projections'):
            if self.projections.shape[1] != len(value):
                raise ValueError("Number of projections does not match"
                                 " projection geometry.")
            if (self.projections.shape[0] != value[0].detector.rows
                or self.projections.shape[2] != value[0].detector.cols):
                raise ValueError("Projection shape does not match projection"
                                 " geometry shape.")

        self._proj_geom = value
        det_shape = (self._proj_geom[0].detector.rows,
                     self._proj_geom[0].detector.cols)
        vectors = self._geoms2vectors(value)
        self._astra_proj_geom = astra.create_proj_geom(
            'cone_vec', *det_shape, vectors)

    @projection_geometry.deleter
    def projection_geometry(self):
        del self._proj_geom
        del self._astra_proj_geom  # TODO: does this delete?
        try:
            del self._astra_projector_id  # could not be generated/valid
        except AttributeError:
            pass

    @staticmethod
    def _geoms2vectors(geoms: List[ProjectionGeometry]):
        """Convert a list of geometries to ASTRA vectors."""

        def geom2astra(g):
            c = lambda x: (x[1], x[0], x[2])
            d = lambda x: np.array((x[1], x[0], x[2])) * g.detector.pixel_width
            e = lambda x: np.array(
                (x[1], x[0], x[2])) * g.detector.pixel_height
            return [*c(g.source_position), *c(g.detector_position),
                    *d(g.u), *e(g.v)]

        vectors = np.zeros((len(geoms), 12))
        for i, g in enumerate(geoms):
            vectors[i] = geom2astra(g)
        return vectors

    def projector3d(self):
        if not hasattr(self, '_astra_projector_id'):
            if self._astra_proj_geom is None:
                raise RuntimeError("Projection geometries must be given before"
                                   " creating a projector. Use the "
                                   "`projection_geometry` setter of this"
                                   " class.")
            if self._astra_vol_geom is None:
                raise RuntimeError("`VolumeGeometry` must be given before "
                                   "creating a projector. Use the "
                                   "`volume_geometry` setter of this class")
            proj_cfg = {'type': 'cuda3d',
                        'VolumeGeometry': self._astra_vol_geom,
                        'ProjectionGeometry': self._astra_proj_geom,
                        'options': {}}
            self._astra_projector_id = astra.projector3d.create(proj_cfg)
        return self._astra_projector_id


class CompatConeProjector(CompatProjector):
    def __call__(self, additive=False):
        assert hasattr(self, '_astra_vol_link')
        assert hasattr(self, '_astra_proj_link')
        astra.experimental.direct_FPBP3D(
            self.projector3d(),
            self._astra_vol_link,
            self._astra_proj_link,
            int(not additive),
            "FP")
        return self._projections


class CompatConeBackprojector(CompatProjector):
    def __call__(self, additive=False):
        assert hasattr(self, '_astra_vol_link')
        assert hasattr(self, '_astra_proj_link')
        astra.experimental.direct_FPBP3D(
            self.projector3d(),
            self._astra_vol_link,
            self._astra_proj_link,
            int(not additive),
            "BP")
        return self._volume