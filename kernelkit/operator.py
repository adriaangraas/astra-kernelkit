from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple
import cupy as cp

from kernelkit.geom.vol import VolumeGeometry
from kernelkit.geom.proj import Beam, ProjectionGeometry
from kernelkit.projector import (BaseProjector, ConeProjector, ConeBackprojector)


class BaseOperator(ABC):
    """A mathematical operator."""

    @abstractmethod
    def __call__(self, input, out=None):
        raise NotImplementedError

    @property
    def T(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def domain_shape(self) -> Tuple:
        raise NotImplementedError

    @property
    @abstractmethod
    def range_shape(self) -> Tuple:
        raise NotImplementedError


class ProjectorOperator(BaseOperator, ABC):
    """An operator using a forward projection and backprojection."""

    def __init__(self,
                 projection_geometry: List[ProjectionGeometry],
                 volume_geometry: VolumeGeometry,
                 projector: BaseProjector,
                 backprojector: BaseProjector):
        self.projector = projector
        self.backprojector = backprojector
        self.projector.projection_geometry = projection_geometry
        self.projector.volume_geometry = volume_geometry

    @property
    def range_shape(self) -> Tuple:
        nr_angles = len(self.projector.projection_geometry)
        det_shape = (self.projector.projection_geometry[0].detector.rows,
                     self.projector.projection_geometry[0].detector.cols)
        proj_axs = self.projector.projection_axes
        return tuple((nr_angles, *det_shape)[i] for i in proj_axs)

    @property
    def domain_shape(self) -> Tuple:
        vol_axs = self.projector.volume_axes
        return tuple(self.projector.volume_geometry.shape[i]
                     for i in vol_axs)

    def __call__(self, input: cp.ndarray, out: Optional[cp.ndarray] = None):
        """Project a volume onto a set of projections."""
        if input.shape != self.domain_shape:
            raise ValueError(f"Input shape {input.shape} does not match "
                             f"domain shape {self.domain_shape} of "
                             f"operator.")
        if out is None:
            out = cp.zeros(self.range_shape, dtype=cp.float32)
        self.projector.volume = input
        self.projector.projections = out
        return self.projector()

    @property
    def T(self):
        self_ = self

        class _Adjoint(BaseOperator):
            def __init__(self):
                self.projector = self_.backprojector
                self.projector.projection_geometry = (
                    self_.projector.projection_geometry)
                self.projector.volume_geometry = (
                    self_.projector.volume_geometry)

            @property
            def domain_shape(self) -> Tuple:
                return self_.range_shape

            @property
            def range_shape(self) -> Tuple:
                return self_.domain_shape

            def __call__(self, input: cp.ndarray, out=None):
                """Backproject a set of projections into a volume."""
                if input.shape != self.domain_shape:
                    raise ValueError(f"Input shape {input.shape} does not "
                                     f"match domain shape "
                                     f"{self.domain_shape} of operator.")

                if out is None:
                    out = cp.zeros(self.range_shape, dtype=cp.float32)
                self.projector.projections = input
                self.projector.volume = out
                self.projector()
                return out

            @property
            def T(self):
                return self_

        if not hasattr(self, '_T'):
            self._T = _Adjoint()
        return self._T


class XrayTransform(ProjectorOperator):
    """A mathematical operator representing the X-ray transform."""

    def __init__(self,
                 projection_geometry: Any,
                 volume_geometry: Any,
                 volume_axes: Tuple = (0, 1, 2),
                 projection_axes: Tuple = (0, 1, 2),
                 astra_compat: bool = False):
        """Create an X-ray transform operator."""

        for p in projection_geometry:
            if p.beam != Beam.CONE:
                raise NotImplementedError("Only cone beam geometry is "
                                          "supported at the moment.")

        if astra_compat:
            from kernelkit.toolbox import (CompatConeProjector,
                                           CompatConeBackprojector)
            if projection_axes != (1, 0, 2):
                raise ValueError("ASTRA Toolbox compatible projectors can only"
                                 " be used with `projection_axes=(1, 0, 2)`."
                                 " Simply set `projection_axes=(1, 0, 2)` and"
                                 " transpose the input.")
            if volume_axes != (2, 1, 0):
                raise ValueError("ASTRA Toolbox compatible projectors can only"
                                 " be used with `volume_axes=(2, 1, 0)`. "
                                 " Simply set `volume_axes=(2, 1, 0)` and "
                                 " transpose the input.")
            projector = CompatConeProjector()
            backprojector = CompatConeBackprojector()
        else:
            projector = ConeProjector(volume_axes=volume_axes,
                                      projection_axes=projection_axes)
            backprojector = ConeBackprojector(volume_axes=volume_axes,
                                              projection_axes=projection_axes)

        super().__init__(projection_geometry, volume_geometry,
                         projector, backprojector)
