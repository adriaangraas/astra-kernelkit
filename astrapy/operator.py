from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple
import cupy as cp

class Operator(ABC):
    """Projectors that play friendly together as linear operators. Good for
    algorithm design."""

    @abstractmethod
    def __call__(self, input, out=None):
        raise NotImplementedError

    @property
    @abstractmethod
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


class ConebeamTransform(Operator):
    def __init__(self,
                 projection_geometry: Any,
                 volume_geometry: Any,
                 volume_axes: Tuple = (0, 1, 2),
                 projection_axes: Tuple = (0, 1, 2),
                 fp: type = None,
                 bp: type = None,
                 astra_compat: bool = False):
        if astra_compat:
            if projection_axes != (1, 0, 2):
                raise ValueError("ASTRA Toolbox compatible format can only"
                                 " be used with `projection_axes=(1, 0, 2)`.")
            if volume_axes != (2, 1, 0):
                raise ValueError("ASTRA Toolbox compatible format can only"
                                 " be used with `volume_axes=(2, 1, 0)`.")
            self.projector = fp()
        else:
            self.projector = ConeProjector()
        self.projector.projection_geometry = projection_geometry
        self.projector.volume_geometry = volume_geometry
        self._proj_axs = projection_axes
        self._vol_axs = volume_axes
        self._T = None

    @property
    def range_shape(self) -> Tuple:
        nr_angles = len(self.projector.projection_geometry)
        det_shape = (self.projector.projection_geometry[0].detector.rows,
                     self.projector.projection_geometry[0].detector.cols)
        return tuple((nr_angles, *det_shape)[i] for i in self._proj_axs)

    @property
    def domain_shape(self) -> Tuple:
        return tuple(self.projector.volume_geometry.shape[i]
                     for i in self._vol_axs)

    def __call__(self, input: cp.ndarray, out: Optional[cp.ndarray] = None):
        """Project a volume onto a set of projections."""
        if out is None:
            out = cp.zeros(self.range_shape, dtype=cp.float32)
        self.projector.volume = input
        self.projector.projections = out
        self.projector()
        return out

    @property
    def T(self):
        self_ = self

        class _Adjoint(Operator):
            def __init__(self):
                if isinstance(self_.projector, AstraCompatConeProjector):
                    self.projector = AstraCompatConeBackprojector()
                else:
                    self.projector = ConeBackprojector()
                self.projector.projection_geometry = self_.projector. \
                    projection_geometry
                self.projector.volume_geometry = self_.projector. \
                    volume_geometry

            @property
            def domain_shape(self) -> Tuple:
                return self_.range_shape

            @property
            def range_shape(self) -> Tuple:
                return self_.domain_shape

            def __call__(self, input: cp.ndarray, out=None):
                """Backproject a set of projections into a volume."""
                if out is None:
                    out = cp.zeros(self.range_shape, dtype=cp.float32)
                self.projector.projections = input
                self.projector.volume = out
                self.projector()
                return out

            @property
            def T(self):
                return self_

        if self._T is None:
            self._T = _Adjoint()
        return self._T
