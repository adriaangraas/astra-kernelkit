import ctypes
import os
import warnings
from typing import Any
from abc import ABC, abstractmethod

import cupy as cp
import numpy as np

from kernelkit.data import aspitched, ispitched
from kernelkit.geom.proj import GeometrySequence, ProjectionGeometry
from kernelkit.geom.vol import VolumeGeometry
from kernelkit.kernel import copy_to_texture, BaseKernel
from kernelkit.kernels.fan_bp import VoxelDrivenFanBP
from kernelkit.projector import (
    ProjectionsNotSetError,
    Projector,
    VolumeGeometryNotSetError,
    ProjectionGeometryNotSetError,
)


class BackProjector2D(Projector):
    """Backprojector for conebeam geometry"""

    TEXTURE_TYPES = ("array", "pitch2d")

    def __init__(
        self, kernel: VoxelDrivenFanBP = None, texture_type="array", **kernel_kwargs
    ):
        """Constructor for the conebeam backprojector.

        Parameters
        ----------
        kernel : VoxelDrivenConeBP, optional
            The kernel to use for backprojection. If `None`, the default kernel
            is used.
        texture_type : str, optional
            The type of texture to use for the projection data. The default is
            'array'. If 'pitch2d', the projection data is copied into
            a pitched GPU array. Creation of 'pitch2d' is faster, but lookup
            in 'array' texture is faster.
        """
        K = VoxelDrivenFanBP(**kernel_kwargs) if kernel is None else kernel
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
            "pitch2d": self._K.TextureFetching.Tex2D,
            "array": self._K.TextureFetching.Tex2D,
        }
        if not self._K.is_compiled:
            self._K.compile(texture=type2texture[self._texture_type])

        self._params: cp.ndarray
        self._textures: cp.cuda.TextureObject | list[cp.cuda.TextureObject]
        # references to texture objects must be kept in scope:
        self._texture_2d_objects: list[cp.cuda.TextureObject]
        self._vol: Any

    @Projector.volume_geometry.setter
    def volume_geometry(self, value: VolumeGeometry):
        if not value.has_isotropic_voxels():
            raise NotImplementedError(
                f"`{self.__class__.__name__}` is not tested with anisotropic "
                "voxels yet."
            )
        Projector.volume_geometry.fset(self, value)
        try:
            self._params = self._K.geoms2params(
                self._proj_geom_seq, self.volume_geometry
            )
            self._K.set_params(self._params)
        except ProjectionGeometryNotSetError:
            pass

    @Projector.projection_geometry.setter
    def projection_geometry(self, value: list[ProjectionGeometry]):
        Projector.projection_geometry.fset(self, value)
        try:
            self._params = self._K.geoms2params(
                self._proj_geom_seq, self.volume_geometry
            )
            self._K.set_params(self._params)
        except VolumeGeometryNotSetError:
            pass

    @property
    def volume(self) -> cp.ndarray:
        """Returns the reconstruction volume."""
        try:
            return self._vol
        except AttributeError:
            raise AttributeError(
                "No volume set. Please set the volume before calling the backprojector."
            )

    @volume.setter
    def volume(self, value: cp.ndarray):
        """Sets the reconstruction volume."""
        if hasattr(self, "volume_geometry"):
            expected_shp = (self._vol_geom.shape[i] for i in self._K.volume_axes)
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
            raise ProjectionsNotSetError("No projections set in the projector.")

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
                    copy_to_texture(cp.asarray(p), self._texture_type) for p in value
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
                    with cp.cuda.Stream(non_blocking=True):
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
        A first call to this method may capture a CUDA graph, which will be
        launched on subsequent calls. This is only possible if the CUDA graph
        capture mode is enabled (see `cupy.cuda.set_graph_capture_mode`).

        The current implementation only supports output volumes of axes
        (0, 1, 2) or (0, 2, 1). The reason is that, for coalesced writes in
        global memory, the fastest varying dimension should be the last one.
        """
        if not additive:
            self.volume.fill(0.0)
        self._K(self._textures, self._vol, self.volume_geometry)
        if self._K.volume_axes == (1, 0):
            pass
        elif self._K.volume_axes == (0, 1):
            # returning a view (an array transpose)
            # TODO(Adriaan): think about a more sustainable solution
            self._vol[...] = cp.reshape(self._vol, tuple(reversed(self._vol.shape))).T
        else:
            raise NotImplementedError(
                "Sorry! Not yet implemented, but should"
                " be easy enough to do so. Please open an issue."
            )
