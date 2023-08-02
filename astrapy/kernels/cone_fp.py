import copy
from math import ceil
from typing import Any, Sequence, Sized, Tuple

from astrapy.geom import normalize_
from astrapy.geom.proj import GeometrySequence
from astrapy.geom.vol import VolumeGeometry
from astrapy.kernel import Kernel, copy_to_symbol, texture_shape
import numpy as np
import cupy as cp


class ConeProjection(Kernel):
    """Conebeam forward projection kernel."""

    SLICES_PER_THREAD = 16
    PIXELS_IN_X_BLOCK = 32
    PIXELS_PER_Y_THREAD = 32

    def __init__(self,
                 slices_per_thread: int = None,
                 pixels_per_thread: int = None,
                 pixels_in_x_block: int = None,
                 volume_axes: Tuple = (0, 1, 2),
                 projection_axes: Tuple = (0, 1, 2),
                 *args):
        """Conebeam forward projection kernel.

        Parameters
        ----------
        slices_per_thread : int, optional
            Number of slices computed in one thread.
            If `None` defaults to `SLICES_PER_THREAD`.
        pixels_per_thread : int, optional
            Number of pixels computed in one thread.
            If `None` defaults to `PIXELS_PER_Y_THREAD`.
        pixels_in_x_block : int, optional
            Number of pixels computed in one block.
            If `None` defaults to `PIXELS_IN_X_BLOCK`.
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
        """
        self._slices_per_thread = (
            slices_per_thread if slices_per_thread is not None
            else self.SLICES_PER_THREAD)
        self._pixels_per_y_thread = (
            pixels_per_thread if pixels_per_thread is not None
            else self.PIXELS_PER_Y_THREAD)
        self._pixels_in_x_block = (
            pixels_in_x_block if pixels_in_x_block is not None
            else self.PIXELS_IN_X_BLOCK)
        self._vol_axs = volume_axes
        self._proj_axs = projection_axes
        super().__init__('cone_fp.cu', *args)

    @property
    def volume_axes(self):
        return self._vol_axs

    @property
    def projection_axes(self):
        return self._proj_axs

    @staticmethod
    def _get_names():
        return (f'cone_fp<DIR_X>', f'cone_fp<DIR_Y>', f'cone_fp<DIR_Z>')

    def compile(self, nr_projs):
        return self._compile(
            names=self._get_names(),
            template_kwargs={'slices_per_thread': self.SLICES_PER_THREAD,
                             'pixels_per_thread': self.PIXELS_PER_Y_THREAD,
                             'nr_projs_global': nr_projs,
                             'volume_axes': self._vol_axs,
                             'projection_axes': self._proj_axs, })

    def __call__(
        self,
        volume_texture: cp.cuda.texture.TextureObject,
        volume_geometry: VolumeGeometry,
        projection_geometry: Sequence,
        projections: Sized,
        rays_per_pixel: int = 1
    ):
        """Forward projection with conebeam geometry.

        Then a loop is initiated over the angles in the batch. For every new
        angle the major direction is inferred.

        One kernel is launched on a grid of blocks with grid-dimensions:
            (nr-regions, nr-angle-batches)
        with
         - each regions size PIXELS_PER_BLOCK_U * PIXELS_PER_BLOCK_V
         - an angle batch size of MAX_ANGLES.
        So each block
         - owns a small portion of the detector;
         - owns a consecutive sequence of projection angles.
        and runs a thread for each combination (pixel-u, projection-angle).
        Then that thread loops through a number on pixels in the row.

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
            Projection data. Either provided as a list of 2D arrays or
            as a single 3D array. The shape of the array-like object must be
            according to the `projection_axes` specified in the constructor.
        rays_per_pixel : int, optional
            Supersampling is currently not supported.
        """
        if isinstance(projection_geometry, list):
            projection_geometry = GeometrySequence.fromList(
                projection_geometry)
        else:
            projection_geometry = copy.deepcopy(projection_geometry)

        volume_shape = volume_geometry.shape
        if not volume_geometry.has_isotropic_voxels():
            raise NotImplementedError(
                f"`{self.__class__.__name__}` is not tested with anisotropic "
                f"voxels yet.")

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
                    f"The length of projection array must match the number of "
                    f" {err} in the projection geometry, as the first axis "
                    f" `projection_axes`, {self._proj_axs[0]}, is used.")

            # TODO(Adriaan): Checking other axes is hard, because looping the
            #                projections and validating the row/column shapes
            #                is not efficient. This is probably because D2H
            #                transfers are incurred when checking a CuPy array
            #                shape.

            for arr in projections:  # regardless of axes format used
                if not arr.flags['C_CONTIGUOUS']:
                    raise ValueError(
                        f"Projection data must be C-contiguous, but got "
                        f"non-contiguous array.")
                if not isinstance(arr, cp.ndarray):
                    raise TypeError(
                        f"Projection data must be of type `cupy.ndarray`, but "
                        f"got {type(arr)}.")
                if arr.dtype not in self.SUPPORTED_DTYPES:
                    raise NotImplementedError(
                        f"Currently there is only support for dtypes "
                        f"{self.SUPPORTED_DTYPES}, but got {arr.dtype}.")

        if rays_per_pixel != 1:
            raise NotImplementedError(
                "Detector supersampling is currently not supported.")

        # TODO: do not do D2H here to support CUDA graphs
        proj_ptrs = [p.data.ptr for p in projections]
        assert len(proj_ptrs) == len(set(proj_ptrs)), (
            "All projection files need to reside in own memory.")
        proj_ptrs = cp.array(proj_ptrs)

        normalize_(projection_geometry, volume_geometry)

        module = self.compile(len(projection_geometry))
        funcs = [module.get_function(n) for n in self._get_names()]
        self._upload_geometries(projection_geometry, module)
        output_scale = self._output_scale(volume_geometry.voxel_size)
        rows = int(projection_geometry.detector.rows[0])
        cols = int(projection_geometry.detector.cols[0])
        blocks_x = ceil(rows / self._pixels_in_x_block)
        blocks_y = ceil(cols / self._pixels_per_y_thread)

        def _launch(start: int, stop: int, axis: int):
            # print(f"Launching {start}-{stop}, axis {axis}")
            # slice volume in the axis of the ray direction
            for i in range(0, volume_shape[axis], self._slices_per_thread):
                funcs[axis](
                    (blocks_x, blocks_y, stop - start),  # grid
                    (self._pixels_in_x_block,),  # threads
                    (volume_texture,
                     proj_ptrs,
                     i,  # start slice
                     start,  # projection
                     *volume_shape,
                     len(projection_geometry), rows, cols,
                     *cp.float32(output_scale[axis])))

        # directions of ray for each projection (a number: 0, 1, 2)
        if projection_geometry.xp == np:
            geom_axis = np.argmax(np.abs(
                projection_geometry.source_position - projection_geometry.detector_position),
                axis=1)
        elif projection_geometry.xp == cp:
            geom_axis = cp.argmax(cp.abs(
                projection_geometry.source_position - projection_geometry.detector_position),
                axis=1, dtype=cp.int32).get()
        else:
            raise Exception("Unknown array module type of geometry.")

        # Run over all angles, grouping them into groups of the same
        # orientation (roughly horizontal vs. roughly vertical).
        # Start a stream of grids for each such group.
        # TODO(Adriaan): sort and launch?
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
        """Returns an output scale for each axis"""
        eps = 1e-4
        v = voxel_size
        if abs(v[0] / v[1] - 1) > eps or abs(v[0] / v[2] - 1) > eps:
            x = [(v[1] / v[0]) ** 2, (v[2] / v[0]) ** 2, v[0] ** 2]
            y = [(v[0] / v[1]) ** 2, (v[2] / v[1]) ** 2, v[0] * v[1]]
            z = [(v[0] / v[2]) ** 2, (v[1] / v[2]) ** 2, v[0] * v[2]]
            output_scale = [x, y, z]
        else:  # cube is almost the same for all directions, thus * 3
            output_scale = [[1., 1., v[0]]] * 3

        return output_scale

    @staticmethod
    def _upload_geometries(
        geometries,
        module: cp.RawModule):
        """Transfer geometries to device as structure of arrays."""
        # TODO(Adriaan): maybe make a mapping between variables
        xp = geometries.xp
        src = xp.ascontiguousarray(geometries.source_position.T)
        ext_min = xp.ascontiguousarray(geometries.detector_extent_min.T)
        u = xp.ascontiguousarray(
            geometries.u.T * geometries.detector.pixel_width)
        v = xp.ascontiguousarray(
            geometries.v.T * geometries.detector.pixel_height)
        srcsX, srcsY, srcsZ = src[0], src[1], src[2]
        detsSX, detsSY, detsSZ = ext_min[0], ext_min[1], ext_min[2]
        dUX, dUY, dUZ = u[0], u[1], u[2]
        dVX, dVY, dVZ = v[0], v[1], v[2]
        copy_to_symbol(module, 'srcsX', srcsX)
        copy_to_symbol(module, 'srcsY', srcsY)
        copy_to_symbol(module, 'srcsZ', srcsZ)
        copy_to_symbol(module, 'detsSX', detsSX)
        copy_to_symbol(module, 'detsSY', detsSY)
        copy_to_symbol(module, 'detsSZ', detsSZ)
        copy_to_symbol(module, 'detsUX', dUX)
        copy_to_symbol(module, 'detsUY', dUY)
        copy_to_symbol(module, 'detsUZ', dUZ)
        copy_to_symbol(module, 'detsVX', dVX)
        copy_to_symbol(module, 'detsVY', dVY)
        copy_to_symbol(module, 'detsVZ', dVZ)
