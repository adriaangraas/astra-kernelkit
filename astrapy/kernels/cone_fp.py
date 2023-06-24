from math import ceil
from typing import Any, Sequence, Sized
from cupy.cuda.texture import TextureObject
from astrapy.data import *
from astrapy.geom import *
from astrapy.kernel import (copy_to_symbol, texture_shape, Kernel)


class ConeProjection(Kernel):
    SLICES_PER_THREAD = 16
    PIXELS_IN_X_BLOCK = 32
    PIXELS_PER_Y_THREAD = 32

    def __init__(self,
                 slices_per_thread: int = None,
                 pixels_per_thread: int = None,
                 pixels_in_x_block: int = None,
                 projs_row_major: bool = True,
                 mode_row: bool = False,
                 *args):
        self._slices_per_thread = (
            slices_per_thread if slices_per_thread is not None
            else self.SLICES_PER_THREAD)
        self._pixels_per_y_thread = (
            pixels_per_thread if pixels_per_thread is not None
            else self.PIXELS_PER_Y_THREAD)
        self._pixels_in_x_block = (
            pixels_in_x_block if pixels_in_x_block is not None
            else self.PIXELS_IN_X_BLOCK)
        self._projs_row_major = projs_row_major
        self._mode_row = mode_row
        super().__init__('cone_fp.cu', *args)

    def _get_names(self):
        return (f'cone_fp<DIR_X>', f'cone_fp<DIR_Y>', f'cone_fp<DIR_Z>')

    def compile(self, nr_projs):
        return self._compile(
            names=self._get_names(),
            template_kwargs={'slices_per_thread': self.SLICES_PER_THREAD,
                             'pixels_per_thread': self.PIXELS_PER_Y_THREAD,
                             'nr_projs_global': nr_projs,
                             'mode_row': self._mode_row,
                             'projs_row_major': self._projs_row_major})

    def __call__(
        self,
        volume_texture: TextureObject,
        volume_extent_min: Sequence,
        volume_extent_max: Sequence,
        geometries: Sequence,
        projections: Sized,
        volume_rotation: Sequence = (0., 0., 0.),
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
        """
        if isinstance(geometries, list):
            geometries = GeometrySequence.fromList(geometries)
        else:
            geometries = copy.deepcopy(geometries)

        # TODO(Adriaan): explicitly formalize that multiple detectors is
        #   problematic atm. Maybe consider more advanced geometries where
        #   the detector is shared.
        #   Note that multiple detectors might not be a problem, they
        #   just render different u's and v's, and if they factually have
        #   less rows and columns, the computation is silently doing
        #   something that is similar to supersampling.
        assert len(projections) == len(geometries)
        for proj, rows, cols in zip(projections, geometries.detector.rows,
                                    geometries.detector.cols):
            if not isinstance(proj, cp.ndarray):
                raise TypeError("`projections` must be a CuPy ndarray.")
            if proj.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for "
                    f"dtype={self.SUPPORTED_DTYPES}.")
            if self._projs_row_major:
                if proj.shape != (rows, cols):
                    raise ValueError("Projection shapes need to "
                                     "match detector (rows, cols) or "
                                     "set `proj_rows_major` to `False`.")
            else:
                if proj.shape != (cols, rows):
                    raise ValueError("Projection shapes need to "
                                     "match detector (cols, rows) or "
                                     "set `projs_rows_major`.")
            if not proj.flags['C_CONTIGUOUS']:
                raise ValueError("Projection data must be C-contiguous in "
                                 "memory.")

        if rays_per_pixel != 1:
            raise NotImplementedError(
                "Detector supersampling is currently not supported.")

        volume_shape = texture_shape(volume_texture)
        if not has_isotropic_voxels(
            volume_shape, volume_extent_min, volume_extent_max):
            raise NotImplementedError(
                f"`{self.__class__.__name__}` is not tested with anisotropic "
                f"voxels yet.")

        proj_ptrs = [p.data.ptr for p in projections]
        assert len(proj_ptrs) == len(set(proj_ptrs)), (
            "All projection files need to reside in own memory.")
        proj_ptrs = cp.array(proj_ptrs)

        vox_size = voxel_size(volume_shape, volume_extent_min,
                              volume_extent_max)
        normalize_geoms_(geometries, volume_extent_min, volume_extent_max,
                         vox_size, volume_rotation)

        module = self.compile(len(geometries))
        funcs = [module.get_function(n) for n in self._get_names()]
        self._upload_geometries(geometries, module)
        output_scale = self._output_scale(vox_size)
        rows = int(geometries.detector.rows[0])
        cols = int(geometries.detector.cols[0])
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
                     rows, cols,
                     *cp.float32(output_scale[axis])))

        # directions of ray for each projection (a number: 0, 1, 2)
        if geometries.xp == np:
            geom_axis = np.argmax(np.abs(
                geometries.tube_position - geometries.detector_position),
                axis=1)
        elif geometries.xp == cp:
            geom_axis = cp.argmax(cp.abs(
                geometries.tube_position - geometries.detector_position),
                axis=1, dtype=cp.int32).get()
        else:
            raise Exception("Geometry computation backend not understood.")

        # Run over all angles, grouping them into groups of the same
        # orientation (roughly horizontal vs. roughly vertical).
        # Start a stream of grids for each such group.
        # TODO(Adriaan): sort and launch?
        start = 0
        for i in range(len(geometries)):
            # if direction changed: launch kernel for this batch
            if geom_axis[i] != geom_axis[start]:
                _launch(start, i, geom_axis[start])
                start = i

        # launch kernel for remaining projections
        _launch(start, len(geometries), geom_axis[start])

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
    def _upload_geometries(geometries: GeometrySequence, module: cp.RawModule):
        """Transfer geometries to device as structure of arrays."""
        # TODO(Adriaan): maybe make a mapping between variables
        xp = geometries.xp
        src = xp.ascontiguousarray(geometries.tube_position.T)
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