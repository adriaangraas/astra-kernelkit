import ctypes
from typing import Any
from abc import ABC, abstractmethod

import cupy as cp
import numpy as np

from kernelkit.data import ispitched, pitched_shape
from kernelkit.geom.proj import Beam, ProjectionGeometrySequence, \
    ProjectionGeometry
from kernelkit.geom.vol import VolumeGeometry
from kernelkit.kernel import copy_to_texture, BaseKernel
from kernelkit.kernels.cone_bp import VoxelDrivenConeBP
from kernelkit.kernels.cone_fp import RayDrivenConeFP


class ProjectionGeometryNotSetError(AttributeError):
    """Raised when the projection geometry is not set in the projector."""

    pass


class VolumeGeometryNotSetError(AttributeError):
    """Raised when the volume geometry is not set in the projector."""

    pass


class VolumeNotSetError(AttributeError):
    """Raised when the volume is not set in the projector."""

    pass


class ProjectionsNotSetError(AttributeError):
    """Raised when the projections are not set in the projector."""

    pass


class BaseProjector(ABC):
    """Interface-like base class for projectors.

    Notes
    -----
    Projectors are optimized for the reuse of geometries and memory for typical
    reconstruction problems, where there is a single set of projections and a
    single reconstruction volume.
    """

    @property
    @abstractmethod
    def projection_axes(self) -> tuple:
        raise NotImplementedError()

    @property
    @abstractmethod
    def volume_axes(self) -> tuple:
        raise NotImplementedError()

    @property
    @abstractmethod
    def projection_geometry(self) -> list[ProjectionGeometry] | None:
        raise NotImplementedError()

    @projection_geometry.setter
    @abstractmethod
    def projection_geometry(self, value: list[ProjectionGeometry]):
        raise NotImplementedError()

    @projection_geometry.deleter
    @abstractmethod
    def projection_geometry(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def volume_geometry(self) -> VolumeGeometry | None:
        raise NotImplementedError()

    @volume_geometry.setter
    @abstractmethod
    def volume_geometry(self, value: VolumeGeometry):
        raise NotImplementedError()

    @volume_geometry.deleter
    @abstractmethod
    def volume_geometry(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def volume(self) -> cp.ndarray:
        raise NotImplementedError()

    @volume.setter
    @abstractmethod
    def volume(self, value: cp.ndarray):
        raise NotImplementedError()

    @volume.deleter
    @abstractmethod
    def volume(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def projections(self) -> cp.ndarray:
        raise NotImplementedError()

    @projections.setter
    @abstractmethod
    def projections(self, value: cp.ndarray):
        raise NotImplementedError()

    @projections.deleter
    @abstractmethod
    def projections(self):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, additive: bool = False):
        raise NotImplementedError()


class Projector(BaseProjector, ABC):
    """Standard projector."""

    def __init__(self, kernel: BaseKernel):
        """Constructor.

        Parameters
        ----------
        kernel : BaseKernel
            The kernel to use for projection or backprojection.
        """
        self._K = kernel
        self._proj_geom: list[ProjectionGeometry]
        self._proj_geom_seq: ProjectionGeometrySequence
        self._vol_geom: VolumeGeometry

    @property
    def projection_axes(self):
        return self._K.projection_axes

    @property
    def volume_axes(self):
        return self._K.volume_axes

    @property
    def projection_geometry(self) -> list[ProjectionGeometry]:
        try:
            return self._proj_geom
        except AttributeError:
            raise ProjectionGeometryNotSetError(
                f"Projection geometry not set in '{self.__class__.__name__}'. "
                "Please set the projection geometry using `projection_geometry`."
            )

    @projection_geometry.setter
    def projection_geometry(self, value: list[ProjectionGeometry]):
        for p in value:
            if p.beam != value[0].beam:
                raise NotImplementedError(
                    "All geometries in the list must have the same beam type."
                )
        self._proj_geom = value

        # convert to ProjectionGeometrySequence for efficient conversion to kernel params
        if not isinstance(value, ProjectionGeometrySequence):
            seq = ProjectionGeometrySequence.fromList(self._proj_geom)
        else:
            seq = value

        self._proj_geom_seq = seq

        try:
            self._params = self._K.geoms2params(
                self._proj_geom_seq, self.volume_geometry
            )
            self._K.set_params(self._params)
        except (VolumeGeometryNotSetError,):
            pass

    @projection_geometry.deleter
    def projection_geometry(self):
        del self._proj_geom
        del self._proj_geom_seq

    @property
    def volume_geometry(self):
        try:
            return self._vol_geom
        except AttributeError:
            raise VolumeGeometryNotSetError(
                f"Volume geometry not set in '{self.__class__.__name__}'. "
                "Please set the volume geometry using `volume_geometry`."
            )

    @volume_geometry.setter
    def volume_geometry(self, value: VolumeGeometry):
        self._vol_geom = value

        try:
            self._params = self._K.geoms2params(
                self._proj_geom_seq, self.volume_geometry
            )
            self._K.set_params(self._params)
        except (ProjectionGeometryNotSetError, AttributeError) as e:
            if e.name != "_proj_geom_seq":
                raise

    @volume_geometry.deleter
    def volume_geometry(self):
        del self._vol_geom


class ForwardProjector(Projector):
    """Cone-beam forward projector."""

    def __init__(
        self,
        kernel: RayDrivenConeFP = None,
        volume_axes=(0, 1, 2),
        projection_axes=(0, 1, 2),
    ):
        """Constructor.

        Parameters
        ----------
        kernel : RayDrivenConeFP, optional
            The kernel to use for projection. If `None`, the default kernel is
            used.
        volume_axes : tuple, optional
            The axes of the volume array that correspond to the x, y, and z
            axes of the volume geometry. The default is (0, 1, 2).
        projection_axes : tuple, optional
            The axes of the projection array or list that correspond to the
            angles, u, and v axes of the projection geometry. The default is
            (0, 1, 2).
        """
        K = (
            RayDrivenConeFP(volume_axes=volume_axes,
                            projection_axes=projection_axes)
            if kernel is None
            else kernel
        )
        super().__init__(K)
        self._projs: cp.ndarray | list[cp.ndarray]
        self._vol_ref: cp.ndarray
        self._texture: cp.cuda.TextureObject

        if not self._K.is_compiled:
            self._K.compile()

    @property
    def volume(self) -> cp.ndarray:
        """Returns the reconstruction volume."""
        try:
            return self._vol_ref
        except AttributeError:
            raise VolumeNotSetError(
                "Volume not set. Use the `volume` setter before calling the projector."
            )

    @volume.setter
    def volume(self, value: cp.ndarray):
        """Sets or updates the reconstruction volume in place.

        Parameters
        ----------
        value : cp.ndarray
            The reconstruction volume. Must be a 3D array with the same shape
            as the volume geometry."""
        if not isinstance(value, cp.ndarray):
            raise TypeError("Volume must be a `cp.ndarray`.")

        self._vol_ref = value  # avoid garbage collection
        if not hasattr(self, "_texture"):
            self._texture = copy_to_texture(value)
        else:
            # TODO(Adriaan): assert shape
            self._texture.ResDesc.cuArr.copy_from(value)

    @volume.deleter
    def volume(self):
        """Deletes the reconstruction volume and frees texture memory."""
        del self._vol_ref
        del self._texture

    @property
    def projections(self) -> cp.ndarray | list[cp.ndarray]:
        try:
            return self._projs
        except AttributeError:
            raise ProjectionsNotSetError(
                "Projections not set. Use the `projections` "
                "setter before calling the projector."
            )

    @projections.setter
    def projections(self, value: cp.ndarray | list[cp.ndarray]):
        """Sets or updates the projection data in place."""
        if not isinstance(value, cp.ndarray) and not isinstance(value, list):
            raise TypeError("Projections must be a `cp.ndarray` or `list`.")

        self._projs = value

        proj_ptrs = [p.data.ptr for p in self._projs]
        assert len(proj_ptrs) == len(
            set(proj_ptrs)
        ), "All projection files need to reside in own memory."

        if not hasattr(self, "_proj_ptrs"):
            self._proj_ptrs = cp.array(proj_ptrs)
        else:
            assert len(proj_ptrs) == len(
                self._proj_ptrs
            ), "Number of projections has changed."
            for i, ptr in enumerate(proj_ptrs):
                self._proj_ptrs[i] = ptr

        self._pitched_proj_shape = None
        if isinstance(self._projs, list):
            if ispitched(self._projs[0]):
                self._pitched_proj_shape = (
                    len(self._projs), *pitched_shape(self._projs[0]))

    def __call__(self, additive: bool = False):
        """Projects the volume into the projections.

        Parameters
        ----------
        additive : bool, optional
            If True, the projections are updated, rather than overwritten.
        """
        if not additive:
            if isinstance(self._projs, cp.ndarray):
                self._projs.fill(0.0)  # much faster than list below
            else:
                # TODO(Adriaan): slow. Faster fill in the kernel?
                # TODO(Adriaan): or try with non-blocking stream
                [p.fill(0.0) for p in self._projs]

        self._K(self._texture, self._proj_geom_seq, self._vol_geom,
                self._proj_ptrs, projections_pitch=self._pitched_proj_shape)


class BackProjector(Projector):
    """Backprojector for conebeam geometry"""

    TEXTURE_TYPES = "array", "layered", "array2d", "pitch2d"

    def __init__(
        self,
        kernel: VoxelDrivenConeBP = None,
        texture_type: str = "layered",
        **kernel_kwargs,
    ):
        """Constructor for the conebeam backprojector.

        Parameters
        ----------
        kernel : VoxelDrivenConeBP, optional
            The kernel to use for backprojection. If `None`, the default kernel
            is used.
        """
        K = VoxelDrivenConeBP(**kernel_kwargs) if kernel is None else kernel
        if kernel is not None and len(kernel_kwargs) > 0:
            raise ValueError(
                "If `kernel` is given, other keyword arguments cannot "
                "be passed to the kernel constructor. You passed "
                f"`{kernel_kwargs}`."
            )
        super().__init__(K)
        self._texture_type = texture_type.lower()
        if self._texture_type not in self.TEXTURE_TYPES:
            raise ValueError(
                "`TEXTURE_TYPE` not supported. Please choose "
                f"one of the following: {self.TEXTURE_TYPES}."
            )
        type2texture = {
            "pitch2d": self._K.InterpolationMethod.Tex2D,
            "array2d": self._K.InterpolationMethod.Tex2D,
            "array": self._K.InterpolationMethod.Tex3D,
            "layered": self._K.InterpolationMethod.Tex2DLayered,
        }
        if not self._K.is_compiled:
            self._K.compile(texture=type2texture[self._texture_type])

        self._params: cp.ndarray
        self._textures: cp.cuda.TextureObject | list[cp.cuda.TextureObject]
        # references to texture objects must be kept in scope:
        self._texture_2d_objects: list[cp.cuda.TextureObject]
        self._vol: Any

    @property
    def volume(self) -> cp.ndarray:
        """Returns the reconstruction volume."""
        try:
            return self._vol
        except AttributeError:
            raise VolumeNotSetError(
                "No volume set. Please set the volume before calling the backprojector."
            )

    @volume.setter
    def volume(self, value: cp.ndarray):
        """Sets the reconstruction volume."""
        if hasattr(self, "volume_geometry"):
            expected_shp = (self._vol_geom.shape[i] for i in
                            self._K.volume_axes)
            if value.shape != tuple(expected_shp):
                raise ValueError(
                    f"Volume shape {value.shape} does not match volume "
                    f"geometry {self._vol_geom.shape} with axes "
                    f"{self._K.volume_axes}."
                )

        self._vol = value

    @volume.deleter
    def volume(self):
        """Deletes the reconstruction volume"""
        del self._vol

    @property
    def projections(self):
        """Returns the projection data.

        Notes
        -----
         - The array may be pitched if `TEXTURE_TYPE` is 'pitch2d'.
        """
        try:
            if self._texture_type == "pitch2d":
                return [t.ResDesc.arr for t in self._texture_2d_objects]
            elif self._texture_type == "array2d":
                return [t.ResDesc.cuArr for t in self._texture_2d_objects]
            elif self._texture_type == "array" or self._texture_type == "layered":
                return self._textures.ResDesc.cuArr
            else:
                raise RuntimeError
        except AttributeError:
            raise ProjectionsNotSetError(
                "No projections set in the projector.")

    @projections.setter
    def projections(self, value: Any):
        """Sets or updates the projection data.

        Parameters
        ----------
        value : np.ndarray or cp.ndarray or list
            The projection data.

        Notes
        -----
         - If the texture type is based on a CUDA Array, updating
        projections does *not* immediately update the texture memory. Instead,
        it invalidates current texture, triggering a new conversion on the
        next `__call__`.
         - If the current stream is capturing, the projections are
        updated in-place. Otherwise, the reference in the class is just
        set to the given value.
         - In principle this function does not allocate memory. However,
        the array is copied to a pitched layout if `TEXTURE_TYPE`
        is 'pitch2d' and the array is not pitched.
        """
        if not (
            isinstance(value, cp.ndarray)
            or isinstance(value, np.ndarray)
            or isinstance(value, list)
        ):
            raise TypeError("Projections must be a `ndarray` or `list`.")

        # if geometry given, check if it matches
        if hasattr(self, "projection_geometry"):
            shp = (
                len(self.projection_geometry),
                self.projection_geometry[0].detector.rows,
                self.projection_geometry[0].detector.cols,
            )
            expected_shp = tuple((shp[i] for i in self._K.projection_axes))
            if isinstance(value, list):
                if len(value) != expected_shp[0]:
                    raise ValueError(
                        f"Projections shape {len(value)} does not match projection"
                        f" geometry {shp} with axes {self._K.projection_axes}."
                    )
                if value[0].shape != expected_shp[1:]:
                    raise ValueError(
                        f"Projections shape {value[0].shape} does not match projection"
                        f" geometry {shp} with axes {self._K.projection_axes}."
                    )
            elif isinstance(value, cp.ndarray):
                if value.shape != expected_shp:
                    raise ValueError(
                        f"Projections shape {value.shape} does not match projection"
                        f" geometry {shp} with axes {self._K.projection_axes}."
                    )

        if isinstance(value, list):
            if self._K.projection_axes[0] != 0:
                raise ValueError(
                    "Can only use a list of projections when projection axis is 0."
                )
        if self._texture_type == "pitch2d":
            assert [
                ispitched(p) for p in value
            ], "Projections must be pitched, use `kernelkit.as_pitched(..., xp=cp)`."
        else:
            if self._texture_type == "layered":
                if self._K.projection_axes[0] != 0:
                    raise ValueError(
                        "Can only use layered texture when "
                        "the first projection axis is 0."
                    )

        # set texture to a single texture object or a cupy array of pointers
        if not hasattr(self, "_textures"):
            if self._texture_type in ("pitch2d", "array2d"):
                # keep a reference to avoid garbage collection
                self._texture_2d_objects = [
                    copy_to_texture(cp.asarray(p), self._texture_type) for p in
                    value
                ]
                txt = cp.array([p.ptr for p in self._texture_2d_objects])
            elif self._texture_type in ("array", "layered"):
                use_layered = self._texture_type == "layered"
                txt = copy_to_texture(value, "array", layered=use_layered)
            else:
                raise ValueError("Unknown texture type.")
            self._textures = txt
        else:  # update CUDA arrays if necessary
            if self._texture_type == "array2d":
                for p, t in zip(value, self._texture_2d_objects):
                    t.ResDesc.cuArr.copy_from(p)
            elif self._texture_type in ("array", "layered"):
                assert value.ndim == 3
                stream = cp.cuda.get_current_stream()
                self._textures.ResDesc.cuArr.copy_from(value, stream)
            elif self._texture_type == "pitch2d":
                # dilemma: updating existing texture, with a device-device
                # copy may be slower than creating new texture. But creating
                # new texture may invalidate a graph?
                for i, p in enumerate(value):
                    if isinstance(p, np.ndarray):
                        addr = p.ctypes.data_as(ctypes.c_void_p)
                    else:
                        addr = p.data.ptr
                    (
                        self._texture_2d_objects[
                            i
                        ].ResDesc.arr.data.copy_from_async(addr, p.nbytes)
                    )

    @projections.deleter
    def projections(self):
        """Deletes the internal texture."""
        for attr in ["_textures", "_texture_2d_objects"]:
            try:
                delattr(self, attr)
            except AttributeError:
                pass  # It's OK if they do not exist

    def __call__(self, additive: bool = False):
        """Backprojects the projections into the volume.

        Parameters
        ----------
        additive : bool, optional
            If True, the volume is updated, rather than overwritten.

        Returns
        -------
        volume : array-like
            The reconstructed volume.

        Notes
        -----
        The current implementation only supports output volumes of axes
        (0, 1, 2) or (0, 2, 1). The reason is that, for coalesced writes in
        global memory, the fastest varying dimension should be the last one.
        """
        if not additive:
            self.volume.fill(0.0)
        self._K(self._textures, self._vol, self.volume_geometry)