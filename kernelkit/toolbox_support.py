from abc import ABC

import astra
import astra.experimental
import cupy as cp
import numpy as np

from kernelkit.projector import (
    BaseProjector,
    ProjectionGeometryNotSetError,
    ProjectionsNotSetError,
    VolumeGeometryNotSetError,
    VolumeNotSetError,
)
from kernelkit.geom.proj import Beam, ProjectionGeometry
from kernelkit.geom.vol import VolumeGeometry


def _geom2astra(g: ProjectionGeometry) -> list[float]:
    """Convert a `kernelkit.ProjectionGeometry` to an ASTRA Toolbox vector."""

    return [
        *g.source_position,
        *g.detector_position,
        *np.asarray(g.u) * g.detector.pixel_width,
        *np.asarray(g.v) * g.detector.pixel_height
    ]


class BaseProjectorAdapter(BaseProjector, ABC):
    """Base class for ASTRA projectors."""

    __slots__ = "_volume", "_projections", "_vol_geom", "_proj_geom"

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
        if hasattr(self, "volume_geometry"):
            expected_shape = tuple(reversed(self.volume_geometry.shape))
            if expected_shape != value.shape:
                raise ValueError(
                    f"Expected volume of shape {expected_shape} (z, y, x)"
                    " according to the volume geometry, but got "
                    f"{value.shape} instead."
                )

        # Make sure CuPy doesn't clean up the volume while ASTRA is busy
        # by keeping a reference to the volume.
        assert value.dtype == cp.float32
        assert value.flags.c_contiguous
        self._volume = value
        z, y, x = self._volume.shape
        # This operation is by my knowledge always efficient, so does not
        # require caching.
        self._astra_vol_link = astra.pythonutils.GPULink(
            self._volume.data.ptr, x, y, z, x * 4
        )

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
        return 2, 1, 0

    @property
    def projections(self) -> cp.ndarray:
        """The projections onto which to project."""
        try:
            return self._projections
        except AttributeError:
            raise ProjectionsNotSetError(
                "Projections must be set before accessing them."
            )

    @projections.setter
    def projections(self, value: cp.ndarray):
        """The projections onto which to project."""
        if not isinstance(value, cp.ndarray):
            raise TypeError("Projections must be a CuPy array.")

        if hasattr(self, "projection_geometry"):
            expected_shape = (
                self.projection_geometry[0].detector.rows,
                len(self.projection_geometry),
                self.projection_geometry[0].detector.cols,
            )
            if value.shape != expected_shape:
                raise ValueError(
                    "Shape does not match projection"
                    f" geometry. Got {value.shape}, "
                    f" expected {expected_shape}."
                )

        assert value.dtype == cp.float32
        assert value.flags.c_contiguous
        z, y, x = value.shape
        self._projections = value
        self._astra_proj_link = astra.pythonutils.GPULink(
            self._projections.data.ptr, x, y, z, x * 4
        )

    @projections.deleter
    def projections(self):
        """Delete the projections and clean up ASTRA projector."""
        del self._projections
        del self._astra_proj_link

    @property
    def projection_axes(self):
        """The axes of the volume to project from."""
        return 1, 0, 2

    @property
    def volume_geometry(self):
        try:
            return self._vol_geom
        except AttributeError:
            raise VolumeGeometryNotSetError(
                f"Volume geometry not set in '{self.__class__.__name__}'. "
                f"Please set the volume geometry using "
                f"`{self.__class__.__name__}.volume_geometry`."
            )

    @volume_geometry.setter
    def volume_geometry(self, value: VolumeGeometry):
        try:
            del self.volume_geometry  # invalidate projector
        except AttributeError:
            pass

        if hasattr(self, "volume"):
            expected_shape_volume = tuple(reversed(self.volume.shape))
            if value.shape != expected_shape_volume:
                raise ValueError(
                    "The given volume geometry does not match the "
                    "shape of the volume in the projector. Got "
                    f"{value.shape} in (x, y, z) convention of the "
                    "volume geometry, but the volume shape is "
                    f"{self.volume.shape} in (z, y, x) convention."
                )

        self._vol_geom = value
        vol_shp = value.shape
        ext_min = value.extent_min
        ext_max = value.extent_max
        self._astra_vol_geom = astra.create_vol_geom(
            vol_shp[1],
            vol_shp[0],
            vol_shp[2],
            ext_min[0],
            ext_max[0],
            ext_min[1],
            ext_max[1],
            ext_min[2],
            ext_max[2],
        )

    @volume_geometry.deleter
    def volume_geometry(self):
        del self._vol_geom
        del self._astra_vol_geom
        try:
            del self._astra_projector_id
        except AttributeError:
            pass

    @property
    def projection_geometry(self) -> list[ProjectionGeometry]:
        try:
            return self._proj_geom
        except AttributeError:
            raise ProjectionGeometryNotSetError(
                f"Projection geometry not set in '{self.__class__.__name__}'. "
                f"Please set the projection geometry using "
                f"`{self.__class__.__name__}.projection_geometry`."
            )

    @projection_geometry.setter
    def projection_geometry(self, value: list[ProjectionGeometry]):
        try:
            del self.projection_geometry  # invalidate projector
        except AttributeError:
            pass

        for p in value:
            if p.beam != Beam.CONE.value:
                raise NotImplementedError(
                    "Only conebeam geometries are supported at the moment."
                )
        if hasattr(self, "projections"):
            if self.projections.shape[1] != len(value):
                raise ValueError(
                    "Number of projections does not match projection geometry."
                )
            if (
                self.projections.shape[0] != value[0].detector.rows
                or self.projections.shape[2] != value[0].detector.cols
            ):
                raise ValueError("Shape does not match projection geometry shape.")

        self._proj_geom = value
        det_shape = (self._proj_geom[0].detector.rows, self._proj_geom[0].detector.cols)
        vectors = self._geoms2vectors(value)
        self._astra_proj_geom = astra.create_proj_geom("cone_vec", *det_shape, vectors)

    @projection_geometry.deleter
    def projection_geometry(self):
        del self._proj_geom
        del self._astra_proj_geom
        try:
            del self._astra_projector_id  # could not be generated/valid
        except AttributeError:
            pass

    @staticmethod
    def _geoms2vectors(geoms: list[ProjectionGeometry]):
        """Convert a list of geometries to ASTRA vectors."""
        return np.array([_geom2astra(g) for g in geoms])

    def projector3d(self):
        """Create an ASTRA Toolbox projector3d object."""
        if not hasattr(self, "_astra_projector_id"):
            if not hasattr(self, "_astra_proj_geom"):
                raise ProjectionGeometryNotSetError(
                    "Projection geometries must be given before"
                    " creating a projector. Use the "
                    "`projection_geometry` setter of this"
                    " class."
                )
            if not hasattr(self, "_astra_vol_geom"):
                raise VolumeGeometryNotSetError(
                    "`VolumeGeometry` must be given before "
                    "creating a projector. Use the "
                    "`volume_geometry` setter of this class"
                )
            proj_cfg = {
                "type": "cuda3d",
                "VolumeGeometry": self._astra_vol_geom,
                "ProjectionGeometry": self._astra_proj_geom,
                "options": {},
            }
            self._astra_projector_id = astra.projector3d.create(proj_cfg)
        return self._astra_projector_id

    def directFPBP(self, mode, additive=False):
        """Perform ASTRA Toolbox forward or backprojection.

        Parameters
        ----------
        mode : str
            Either 'FP' or 'BP'.
        additive : bool, optional
            Whether to perform an additive or a replacement projection.
            Defaults to False.

        Returns
        -------
        array-like
            The projections if `mode` is 'FP', or the volume if `mode` is 'BP'.
        """
        if not hasattr(self, "_astra_vol_link"):
            raise VolumeNotSetError(
                "Volume must be set before performing a forward or backprojection."
            )
        if not hasattr(self, "_astra_proj_link"):
            raise ProjectionsNotSetError(
                "Projections must be set before performing a forward or backprojection."
            )

        if not mode.upper() in ("FP", "BP"):
            raise ValueError("Mode must be either 'FP' or 'BP'.")

        astra.experimental.direct_FPBP3D(
            self.projector3d(),
            self._astra_vol_link,
            self._astra_proj_link,
            int(not additive),
            mode.upper(),
        )
        return self._projections if mode == "FP" else self._volume


class ForwardProjector(BaseProjectorAdapter):
    """Adapter for ASTRA forward projectors."""

    def __call__(self, additive=False):
        """Perform ASTRA Toolbox forward projection.

        Parameters
        ----------
        additive : bool, optional
            Whether to perform an additive or a replacement projection.
            Defaults to False.

        Returns
        -------
        array-like
            The projections.
        """
        super().directFPBP("FP", additive=additive)


class BackProjector(BaseProjectorAdapter):
    """Adapter for ASTRA backprojectors."""

    def __call__(self, additive=False):
        """Perform ASTRA Toolbox backprojection.

        Parameters
        ----------
        additive : bool, optional
            Whether to perform an additive or a replacement projection.
            Defaults to False.

        Returns
        -------
        array-like
            The volume.
        """
        super().directFPBP("BP", additive=additive)
