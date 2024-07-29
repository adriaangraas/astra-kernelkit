import copy
from importlib import resources
from math import ceil
from typing import Sequence

import numpy as np
import cupy as cp

from kernelkit import KERNELKIT_CUDA_SOURCES
from kernelkit.geom import normalize_
from kernelkit.geom.proj import ProjectionGeometrySequence
from kernelkit.geom.vol import VolumeGeometry
from kernelkit.kernel import (BaseKernel, copy_to_symbol,
                              KernelMemoryUnpreparedException,
                              KernelNotCompiledException)


class RayDrivenConeFP(BaseKernel):
    """Ray-driven conebeam forward projection kernel."""

    FLOAT_DTYPE = cp.float32
    SUPPORTED_DTYPES = [cp.float32]

    slices_per_thread = 16
    pixels_in_x_block = 32
    pixels_in_y_thread = 32

    def __init__(
        self,
        pixels_per_thread: tuple = None,
        slices_per_block: int = None,
        volume_axes: tuple = (0, 1, 2),
        projection_axes: tuple = (0, 1, 2),
    ):
        """Conebeam forward projection kernel.

        Parameters
        ----------
        pixels_per_thread : tuple, optional
            Number of pixels per block in the horizontal and vertical
            directions. Defaults to `(32, 32)`.
        slices_per_block : int, optional
            Number of slices per block in the direction of the ray. Defaults
            to `16`.
        volume_axes : tuple, optional
            Axes of the reconstruction volume. Defaults to `(0, 1, 2)`. The
            axes correspond to 'x', 'y', and 'z', respectively. 'x' is the
            source-detector axis, 'y' the orthogonal axis in the horizontal
            plane, and 'z' the vertical axis. ASTRA Toolbox uses `(2, 1, 0)`.
        projection_axes : tuple, optional
            Axes of the projection data. Defaults to `(0, 1, 2)`. Axis 0
            corresponds to the projection angle, axis 1 to the detector
            column, and axis 2 to the detector row. ASTRA Toolbox uses
            `(1, 0, 2)` for this parameter.
        *args
            Arguments passed to the `Kernel` constructor.

        See Also
        --------
        kernelkit.kernel.BaseKernel : Base class for CUDA kernels.
        """
        if pixels_per_thread is not None:
            self.pixels_per_block = pixels_per_thread
        if slices_per_block is not None:
            self.slices_per_block = slices_per_block
        self._vol_axs = volume_axes
        self._proj_axs = projection_axes
        self._params_are_set: bool = False

        cuda_source = (
            resources
            .files(KERNELKIT_CUDA_SOURCES)
            .joinpath("cone_fp.cu")
            .read_text()
        )
        super().__init__(cuda_source)

    @property
    def volume_axes(self) -> tuple:
        return self._vol_axs

    @property
    def projection_axes(self) -> tuple:
        return self._proj_axs

    @staticmethod
    def _get_names() -> tuple:
        return f"cone_fp<DIR_X>", f"cone_fp<DIR_Y>", f"cone_fp<DIR_Z>"

    def compile(self,
                max_projs: int = 1024):
        return self._compile(
            name_expressions=self._get_names(),
            template_kwargs={
                "slices_per_thread": self.slices_per_thread,
                "pixels_per_thread": self.pixels_in_y_thread,
                "nr_projs_global": max_projs,
                "volume_axes": self._vol_axs,
                "projection_axes": self._proj_axs,
            },
        )

    def __call__(
        self,
        volume_texture: cp.cuda.texture.TextureObject,
        projection_geometry: ProjectionGeometrySequence,
        volume_geometry: VolumeGeometry,
        projections,
        projections_pitch: tuple | None = None
    ):
        """Forward projection with conebeam geometry.

        Parameters
        ----------
        volume_texture : cupy.cuda.texture.TextureObject
            3D texture object with the reconstruction volume in the axes
            specified by `volume_axes` in the constructor.
        volume_geometry : VolumeGeometry
            Geometry of the reconstruction volume.
        projection_geometry : Sequence of ProjectionGeometry
            Geometry of the projection data.
        projections : Sequence of cupy.ndarray or cp.ndarray
            data. Either provided as a list of 2D arrays or
            as a single 3D array. The shape of the array-like object must be
            according to the `projection_axes` specified in the constructor.
        projections_pitch : tuple, optional
            Dimensions tuple of the pitched memory. Defaults to `None`, meaning
            that the dimensions of the projections and detector are used. Note
            the first dimension is the pitch of projections, the second the
            pitch of rows, and the third the pitch of columns, even though
            the projections may be stored in a different axis order. E.g.,
            if projection axis is `(0, 2, 1)`, the pitched dimensions must be
            `(angles, cols + padding, rows)`.
        """
        assert volume_texture is not None
        if isinstance(projections, list):
            if self._proj_axs[0] == 0:
                len_dim_0 = len(projection_geometry)
                err = "projections"
            elif self._proj_axs[1] == 0:
                len_dim_0 = projection_geometry[0].detector.rows
                err = "detector rows"
            else:
                len_dim_0 = projection_geometry[0].detector.cols
                err = "detector columns"
            if len(projections) != len_dim_0:
                raise ValueError(
                    "The length of projection array must match the number of "
                    f" {err} in the projection geometry, as the first axis of "
                    " `projection_axes`. However, "
                    f"{self._proj_axs[0]}, is used for the projection angle."
                )

            # TODO(Adriaan): Checking other axes is hard, because looping the
            #                projections and validating the row/column shapes
            #                is not efficient.
            for arr in projections:  # regardless of axes format used
                if not arr.flags["C_CONTIGUOUS"]:
                    raise ValueError(
                        f"Projection data must be C-contiguous, but got "
                        f"non-contiguous array."
                    )
                if not isinstance(arr, cp.ndarray):
                    raise TypeError(
                        "Projection data must be of type `cupy.ndarray`, but "
                        f"got {type(arr)}."
                    )
                if arr.dtype not in self.SUPPORTED_DTYPES:
                    raise NotImplementedError(
                        "Currently there is only support for dtypes "
                        f"{self.SUPPORTED_DTYPES}, but got {arr.dtype}."
                    )

        if not self.is_compiled:
            raise KernelNotCompiledException(
                "Please compile the kernel with `compile()`."
            )
        funcs = [self._module.get_function(n) for n in self._get_names()]

        if not self._params_are_set:
            raise KernelMemoryUnpreparedException(
                "Please set the projection geometry with `set_params()`."
            )

        output_scale = self._output_scale(volume_geometry.voxel_size)

        rows = int(projection_geometry.detector.rows[0])
        cols = int(projection_geometry.detector.cols[0])
        blocks_x = ceil(rows / self.pixels_in_x_block)
        blocks_y = ceil(cols / self.pixels_in_y_thread)
        volume_shape = volume_geometry.shape

        if projections_pitch is None:
            projections_pitch = (len(projections), rows, cols)


        def _launch(start: int, stop: int, axis: int):
            """Launch kernel for a range of projections"""
            # slice volume in the axis of the ray direction
            for i in range(0, volume_shape[axis], self.slices_per_thread):
                assert projections.dtype == cp.int64
                funcs[axis](
                    (blocks_x, blocks_y, stop - start),  # grid
                    (self.pixels_in_x_block,),  # threads
                    (
                        volume_texture,
                        projections,
                        i,  # start slice
                        start,  # projection
                        *volume_shape,
                        len(projection_geometry),
                        rows,
                        cols,
                        *projections_pitch,
                        *cp.float32(output_scale[axis]),
                    ),
                )

        # directions of ray for each projection (a number: 0, 1, 2)
        if projection_geometry.xp == np:
            geom_axis = np.argmax(
                np.abs(
                    projection_geometry.source_position
                    - projection_geometry.detector_position
                ),
                axis=1,
            )
        elif projection_geometry.xp == cp:
            geom_axis = cp.argmax(
                cp.abs(
                    projection_geometry.source_position
                    - projection_geometry.detector_position
                ),
                axis=1,
                dtype=cp.int32,
            ).get()
        else:
            raise Exception("Unknown array module type of geometry.")

        # Run over all angles, grouping them into groups of the same
        # orientation (roughly horizontal vs. roughly vertical).
        # Start a stream of grids for each such group.
        # TODO(Adriaan): instead sort and launch?
        start = 0
        for i in range(len(projection_geometry)):
            # if direction changed: launch kernel for this batch
            if geom_axis[i] != geom_axis[start]:
                _launch(start, i, geom_axis[start])
                start = i

        # launch kernel for remaining projections
        _launch(start, len(projection_geometry), geom_axis[start])

    @staticmethod
    def _output_scale(voxel_size: np.ndarray) -> list:
        """Returns an output scale for each axis."""
        EPS = 1e-4
        v = voxel_size
        if abs(v[0] / v[1] - 1) > EPS or abs(v[0] / v[2] - 1) > EPS:
            x = [(v[1] / v[0]) ** 2, (v[2] / v[0]) ** 2, v[0] ** 2]
            y = [(v[0] / v[1]) ** 2, (v[2] / v[1]) ** 2, v[0] * v[1]]
            z = [(v[0] / v[2]) ** 2, (v[1] / v[2]) ** 2, v[0] * v[2]]
            output_scale = [x, y, z]
        else:  # cube is almost the same for all directions, thus * 3
            output_scale = [[1.0, 1.0, v[0]]] * 3

        return output_scale

    @staticmethod
    def geoms2params(
        projection_geometry: Sequence[ProjectionGeometrySequence],
        volume_geometry: VolumeGeometry,
    ):
        """Converts geometries to kernel parameters.

        Parameters
        ----------
        pg : Sequence of ProjectionGeometry
            Geometry of the projection data.
        volume_geometry : VolumeGeometry
            Geometry of the reconstruction volume.

        Returns
        -------
        tuple
            Parameters for the kernel.
        """
        pg = projection_geometry

        if isinstance(pg, list):
            pg = ProjectionGeometrySequence.fromList(pg)
            xp = pg.xp
        else:
            pg = copy.deepcopy(pg)
            xp = pg.xp

        normalize_(pg, volume_geometry)

        src = xp.ascontiguousarray(pg.source_position.T)
        ext_min = xp.ascontiguousarray(pg.detector_extent_min.T)
        u = xp.ascontiguousarray(
            pg.u.T * pg.detector.pixel_width)
        v = xp.ascontiguousarray(
            pg.v.T * pg.detector.pixel_height)
        return (src[0], src[1], src[2],
                ext_min[0], ext_min[1], ext_min[2],
                u[0], u[1], u[2],
                v[0], v[1], v[2])

    def set_params(self, params):
        """Transfer geometries to device as structure of arrays."""

        # TODO(Adriaan): maybe make a mapping between variables
        assert [len(params[0]) == len(p) for p in params]
        if len(params[0]) > self._compiled_template_kwargs["nr_projs_global"]:
            raise ValueError(
                f"Number of projections, {len(params) // 12}, exceeds the "
                "the maximum of the compiled kernel, namely "
                f"{self._compiled_template_kwargs['nr_projs_global']}. "
                "Please recompile the kernel with a higher "
                "`max_projs`."
            )

        m = self._module
        copy_to_symbol(m, "srcsX", params[0])
        copy_to_symbol(m, "srcsY", params[1])
        copy_to_symbol(m, "srcsZ", params[2])
        copy_to_symbol(m, "detsSX", params[3])
        copy_to_symbol(m, "detsSY", params[4])
        copy_to_symbol(m, "detsSZ", params[5])
        copy_to_symbol(m, "detsUX", params[6])
        copy_to_symbol(m, "detsUY", params[7])
        copy_to_symbol(m, "detsUZ", params[8])
        copy_to_symbol(m, "detsVX", params[9])
        copy_to_symbol(m, "detsVY", params[10])
        copy_to_symbol(m, "detsVZ", params[11])
        self._params_are_set = True