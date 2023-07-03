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

    def __init__(self):
        """Base class for ASTRA projectors."""
        super().__init__()

        self._volume = None
        self._projections = None
        self._vol_geom = None
        self._proj_geom = None
        self._astra_vol_link = None
        self._astra_proj_link = None
        self._astra_vol_geom = None
        self._astra_proj_geom = None
        self._astra_projector_id = None

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
        if self.volume_geometry is not None:
            expected_shape = tuple(reversed(self.volume_geometry.shape))
            if expected_shape != value.shape:
                raise ValueError(
                    f"Expected volume of shape {expected_shape} (z, y, x)"
                    f" according to the volume geometry, but got "
                    f"{value.shape} instead.")

        if self._volume is not None and self._volume.shape != value.shape:
            self._astra_projector_id = None

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
        self._volume = None
        self._astra_vol_link = None  # TODO: does this delete?
        self._astra_projector_id = None

    @property
    def projections(self) -> cp.ndarray:
        """The projections onto which to project."""
        return self._projections

    @projections.setter
    def projections(self, value: cp.ndarray):
        """The projections onto which to project."""
        assert value.dtype == cp.float32
        assert value.flags.c_contiguous

        if self.projection_geometry is not None:
            expected_shape = (self.projection_geometry[0].detector.rows,
                              len(self.projection_geometry),
                              self.projection_geometry[0].detector.cols)
            if value.shape != expected_shape:
                raise ValueError(f"Projection shape does not match projection"
                                 f" geometry. Got {value.shape}, "
                                 f" expected {expected_shape}.")

        if self._projections is not None \
            and self._projections.shape != value.shape:
            self._astra_projector_id = None

        z, y, x = value.shape
        if (self._astra_proj_link is not None
            and self._projections.shape == value.shape):
            # The advantage of maintaining the proj_link, is that it avoids
            # reinitializing the ASTRA projector, which is expensive.
            self._projections[...] = value
        else:
            # Projections need to be a cache attribute to avoid being cleaned
            # up by CuPy.
            self._projections = value
            # This operation is by my knowledge always efficient, so does not
            # require caching.
            self._astra_proj_link = astra.pythonutils.GPULink(
                self._projections.data.ptr, x, y, z, x * 4)

    @projections.deleter
    def projections(self):
        """Delete the projections and clean up ASTRA projector."""
        self._projections = None
        self._astra_proj_link = None
        self._astra_projector_id = None

    @property
    def volume_geometry(self) -> VolumeGeometry:
        return self._vol_geom

    @volume_geometry.setter
    def volume_geometry(self, value: VolumeGeometry):
        del self.volume_geometry  # invalidate projector

        if self.volume is not None:
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
        self._vol_geom = None
        self._astra_vol_geom = None  # TODO: does this delete?
        self._astra_projector_id = None

    @property
    def projection_geometry(self):
        return self._proj_geom

    @projection_geometry.setter
    def projection_geometry(self, value: List[ProjectionGeometry]):
        del self.projection_geometry  # invalidate projector

        if self.projections is not None:
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
        self._proj_geom = None
        self._astra_proj_geom = None  # TODO: does this delete?
        self._astra_projector_id = None

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
        if self._astra_projector_id is None:
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
        astra.experimental.direct_FPBP3D(
            self.projector3d(),
            self._astra_vol_link,
            self._astra_proj_link,
            int(not additive),
            "FP")
        return self._projections


class CompatConeBackprojector(CompatProjector):
    def __call__(self, additive=False):
        astra.experimental.direct_FPBP3D(
            self.projector3d(),
            self._astra_vol_link,
            self._astra_proj_link,
            int(not additive),
            "BP")
        return self._volume
