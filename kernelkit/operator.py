from abc import ABC, abstractmethod
from typing import Any
import cupy as cp

from kernelkit.geom.vol import VolumeGeometry
from kernelkit.geom.proj import ProjectionGeometry
from kernelkit.projector import BaseProjector, ForwardProjector, BackProjector


class BaseOperator(ABC):
    """A mathematical operator."""

    @property
    @abstractmethod
    def domain_shape(self) -> tuple:
        """The shape of the domain of the operator."""
        raise NotImplementedError

    @property
    @abstractmethod
    def range_shape(self) -> tuple:
        """The shape of the range of the operator."""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, input, out=None):
        """Apply the operator to an input, writing the result to an output."""
        raise NotImplementedError

    @property
    def T(self):
        """The corresponding adjoint of the operator."""
        raise NotImplementedError


class ProjectorOperator(BaseOperator, ABC):
    """An operator :math:`A : X \\to Y` using a forward and backprojector."""

    def __init__(self, projector: BaseProjector, backprojector: BaseProjector):
        """Create a projector operator.

        Parameters
        ----------
        projection_geometry : list[ProjectionGeometry]
            The projection geometries.
        volume_geometry : VolumeGeometry
            The volume geometry.
        projector : BaseProjector
            The forward projector.
        backprojector : BaseProjector
            The backprojector.

        Notes
        -----
        Matching of the projectors is not checked proactively.
        """
        self.projector = projector
        self.backprojector = backprojector

    @property
    def domain_shape(self) -> tuple:
        """The domain :math:`\\text{dom}(A)`"""
        vol_axs = self.projector.volume_axes
        return tuple(self.projector.volume_geometry.shape[i] for i in vol_axs)

    @property
    def range_shape(self) -> tuple:
        """The range :math:`\\text{ran}(A)`."""
        nr_angles = len(self.projector.projection_geometry)
        det_shape = (
            self.projector.projection_geometry[0].detector.rows,
            self.projector.projection_geometry[0].detector.cols,
        )
        proj_axs = self.projector.projection_axes
        return tuple((nr_angles, *det_shape)[i] for i in proj_axs)

    def __call__(self, input: cp.ndarray, out: cp.ndarray | None = None,
                 additive=False):
        """Apply the forward projector, :math:`A(x)`.

        Parameters
        ----------
        input : array-like
            The input to the operator.
        out : array-like, optional
            The output of the operator. If not given, a new array is allocated.
        """
        if input.shape != self.domain_shape:
            raise ValueError(
                f"Input shape {input.shape} does not match "
                f"domain shape {self.domain_shape} of "
                "operator."
            )
        if out is None:
            out = cp.zeros(self.range_shape, dtype=cp.float32)

        self.projector.volume = input
        self.projector.projections = out
        self.projector(additive=additive)
        return out

    @property
    def T(self):
        self_ = self

        class _Adjoint(BaseOperator):
            """The adjoint operator :math:`A^* : Y \\to X`."""

            def __init__(self):
                """Create the adjoint operator."""
                self.projector = self_.backprojector
                self.backprojector = self_.projector

            @property
            def domain_shape(self) -> tuple:
                """The domain :math:`\\text{dom}(A^*)=\\text{ran}(A)`."""
                return self_.range_shape

            @property
            def range_shape(self) -> tuple:
                """The range :math:`\\text{ran}(A^*)=\\text{dom}(A)`."""
                return self_.domain_shape

            def __call__(self, input: cp.ndarray, out=None, additive=False):
                """Apply the backprojector, :math:`A^*(y)`."""
                if input.shape != self.domain_shape:
                    raise ValueError(
                        f"Input shape {input.shape} does not "
                        "match domain shape "
                        f"{self.domain_shape} of operator."
                    )
                if out is None:
                    out = cp.zeros(self.range_shape, dtype=cp.float32)
                self.projector.projections = input
                self.projector.volume = out
                self.projector(additive)
                return out

            @property
            def T(self):
                """The adjoint of the adjoint operator, :math:`A^{**} = A`."""
                return self_

        if not hasattr(self, "_T"):
            self._T = _Adjoint()
        return self._T


class XrayTransform(ProjectorOperator):
    """A mathematical operator representing the X-ray transform."""

    def __init__(
        self,
        projection_geometry: Any,
        volume_geometry: Any,
        volume_axes: tuple = (0, 1, 2),
        projection_axes: tuple = (0, 1, 2),
        use_toolbox: bool = False,
        fp_kwargs=None,
        bp_kwargs=None,
    ):
        """Create an X-ray transform operator.

        Parameters
        ----------
        projection_geometry : list[ProjectionGeometry]
            The projection geometries.
        volume_geometry : VolumeGeometry
            The volume geometry.
        volume_axes : tuple, optional
            The axes of the volume to use for the forward projection.
            Default is (0, 1, 2).
        projection_axes : tuple, optional
            The axes of the projections to use for the forward projection.
            Default is (0, 1, 2).
        use_toolbox : bool, optional
            Whether to use ASTRA Toolbox compatible projectors. Default is
            False.

        Raises
        ------
        NotImplementedError : If the projection geometries are not conebeam.

        Notes
        -----
        If `use_toolbox` is True, the `volume_axes` and `projection_axes`
        must be set to (2, 1, 0) and (1, 0, 2), respectively.
        """
        fp_kwargs = fp_kwargs or {}
        bp_kwargs = bp_kwargs or {}
        if use_toolbox:
            import kernelkit.toolbox_support

            if projection_axes != (1, 0, 2):
                raise ValueError(
                    "ASTRA Toolbox compatible projectors can only"
                    " be used with `projection_axes=(1, 0, 2)`."
                    " Simply set `projection_axes=(1, 0, 2)` and"
                    " transpose the input if necessary."
                )
            if volume_axes != (2, 1, 0):
                raise ValueError(
                    "ASTRA Toolbox compatible projectors can only"
                    " be used with `volume_axes=(2, 1, 0)`. "
                    " Simply set `volume_axes=(2, 1, 0)` and "
                    " transpose the input if necessary."
                )
            if fp_kwargs or bp_kwargs:
                raise ValueError(
                    "ASTRA Toolbox projectors do not accept any additional "
                    "`fp_kwargs` or `bp_kwargs`"
                )

            projector = kernelkit.toolbox_support.ForwardProjector()
            backprojector = kernelkit.toolbox_support.BackProjector()
        else:
            projector = ForwardProjector(
                volume_axes=volume_axes,
                projection_axes=projection_axes,
                **fp_kwargs
            )
            backprojector = BackProjector(
                volume_axes=volume_axes,
                projection_axes=projection_axes,
                **bp_kwargs
            )

        projector.volume_geometry = volume_geometry
        projector.projection_geometry = projection_geometry
        backprojector.volume_geometry = volume_geometry
        backprojector.projection_geometry = projection_geometry

        super().__init__(projector, backprojector)
