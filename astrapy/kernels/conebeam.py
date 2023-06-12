from math import ceil
from typing import Any, Sequence, Sized
from cupy.cuda.texture import TextureObject
from astrapy.data import *
from astrapy.geom import *
from astrapy.kernel import (copy_to_symbol, cuda_float4, texture_shape,
                            Kernel)


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
        return (f'cone_fp<DIR_X>',
                f'cone_fp<DIR_Y>',
                f'cone_fp<DIR_Z>')

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


class ConeBackprojection(Kernel):
    # last dim is not number of threads, but volume slab thickness
    VOXELS_PER_BLOCK = (16, 32, 6)
    LIMIT_PROJS_PER_BLOCK = 32
    MIN_LIMIT_PROJS = 1024

    def __init__(self,
                 min_limit_projs: int = None,
                 voxels_per_block: tuple = None,
                 limit_projs_per_block: int = None):
        super().__init__('cone_bp.cu')
        self._min_limit_projs = (min_limit_projs if min_limit_projs is not None
                                 else self.MIN_LIMIT_PROJS)
        self._vox_block = (voxels_per_block if voxels_per_block is not None
                           else self.VOXELS_PER_BLOCK)
        self._limit_projs_per_block = (
            limit_projs_per_block if limit_projs_per_block is not None
            else self.LIMIT_PROJS_PER_BLOCK)

    def compile(self, nr_projs: int = None,
                use_texture_3D: bool = True) -> cp.RawModule:
        if nr_projs is None:
            nr_projs_global = self._min_limit_projs
        else:
            nr_projs_global = np.max((nr_projs, self._min_limit_projs))

        return self._compile(
            names=('cone_bp',),
            template_kwargs={'nr_vxls_block_x': self._vox_block[0],
                             'nr_vxls_block_y': self._vox_block[1],
                             'nr_vxls_block_z': self._vox_block[2],
                             'nr_projs_block': self._limit_projs_per_block,
                             'nr_projs_global': nr_projs_global,
                             'texture3D': use_texture_3D})

    def __call__(self,
                 projections_textures: Any,
                 params: Sequence,
                 volume: cp.ndarray,
                 volume_extent_min: Sequence,
                 volume_extent_max: Sequence,
                 volume_rotation: Sequence = (0., 0., 0.)):
        """Backprojection with conebeam geometry.

        :type projections_textures: object
            If `TextureObject` the kernel compiles with `texture3D` spec,
            and the kernel expects a 3D contiguous texture block and uses
            `tex3D()` to access it.
            If `list[TextureObject]` the kernel compiles to expect an array
            of pointers to 2D texture objects (one for each projection), and
            accesses the textures by a list of pointers.
        """
        if isinstance(volume, cp.ndarray):
            if volume.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for "
                    f"dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`volume` must be a CuPy ndarray.")
        assert volume.flags.c_contiguous is True, (
            f"`{self.__class__.__name__}` is not tested without "
            f"C-contiguous data.")
        if not has_isotropic_voxels(
            volume.shape, volume_extent_min, volume_extent_max):
            raise NotImplementedError(
                f"`{self.__class__.__name__}` is not tested with anisotropic "
                f"voxels yet.")

        if isinstance(projections_textures, list):
            # TODO(Adriaan): this terribly slows down! don't run!
            # for i, p in enumerate(projections_textures):
            #     if (not _texture_shape(p) ==
            #             (geometries.detector.rows[i],
            #              geometries.detector.cols[i])):
            #         raise ValueError("Projection texture resource needs to"
            #                          " match the detector dimensions in the"
            #                          " geometries.")
            compile_use_texture3D = False
            projections = cp.array([p.ptr for p in projections_textures])
        elif isinstance(projections_textures, TextureObject):
            compile_use_texture3D = True
            projections = projections_textures
        else:
            raise ValueError("Please give in a list of 2D TextureObject, or "
                             "one 3D TextureObject.")

        vox_volume = voxel_volume(
            volume.shape, volume_extent_min, volume_extent_max)

        module = self.compile(len(params), compile_use_texture3D)
        cone_bp = module.get_function("cone_bp")
        copy_to_symbol(module, 'params', params.flatten())

        # TODO(Adriaan): better to have SoA instead AoS?
        blocks = np.ceil(np.asarray(volume.shape) / self._vox_block).astype(
            np.int32)
        for start in range(0, len(params), self._limit_projs_per_block):
            cone_bp((blocks[0] * blocks[1], blocks[2]),  # grid
                    (self._vox_block[0], self._vox_block[1]),  # threads
                    (projections,
                     volume,
                     start,
                     len(params),
                     *volume.shape,
                     cp.float32(vox_volume)))

    @staticmethod
    def geoms2params(
        geometries: GeometrySequence,
        vol_shape,
        volume_extent_min,
        volume_extent_max,
        volume_rotation=(0., 0., 0.)):
        """Precomputed kernel parameters

        We need three things in the kernel:
         - projected coordinates of pixels on the detector:
          u: || (x-s) v (s-d) || / || u v (s-x) ||
          v: -|| u (x-s) (s-d) || / || u v (s-x) ||
         - ray density weighting factor for the adjoint
          || u v (s-d) ||^2 / ( |cross(u,v)| * || u v (s-x) ||^2 )
         - FDK weighting factor
          ( || u v s || / || u v (s-x) || ) ^ 2

        Since u and v are ratios with the same denominator, we have
        a degree of freedom to scale the denominator. We use that to make
        the square of the denominator equal to the relevant weighting factor.

        For FDK weighting:
            goal: 1/fDen^2 = || u v (s-d) ||^2 / ( |cross(u,v)| * || u v (s-x) ||^2 )
            fDen = ( sqrt(|cross(u,v)|) * || u v (s-x) || ) / || u v (s-d) ||
            i.e. scale = sqrt(|cross(u,v)|) * / || u v (s-d) ||
        Otherwise:
            goal: 1/fDen = || u v s || / || u v (s-x) ||
            fDen = || u v (s-x) || / || u v s ||
            i.e., scale = 1 / || u v s ||
        """
        if isinstance(geometries, list):
            geometries = GeometrySequence.fromList(geometries)
        else:
            geometries = copy.deepcopy(geometries)

        xp = geometries.xp

        vox_size = voxel_size(
            vol_shape, volume_extent_min, volume_extent_max)
        normalize_geoms_(geometries, volume_extent_min, volume_extent_max,
                         vox_size, volume_rotation)

        u = geometries.u * geometries.detector.pixel_width[..., xp.newaxis]
        v = geometries.v * geometries.detector.pixel_height[..., xp.newaxis]
        s = geometries.tube_position
        d = geometries.detector_extent_min

        # NB(ASTRA): for cross(u,v) we invert the volume scaling (for the voxel
        # size normalization) to get the proper dimensions for
        # the scaling of the adjoint
        cr = xp.cross(u, v)  # maintain f32
        cr *= xp.array([vox_size[1] * vox_size[2],
                        vox_size[0] * vox_size[2],
                        vox_size[0] * vox_size[1]])
        scale = (xp.sqrt(xp.linalg.norm(cr, axis=1)) /
                 xp.linalg.det(xp.asarray((u, v, d - s)).swapaxes(0, 1)))

        # TODO(Adriaan): it looks like my preweighting is different to ASTRA's
        #   and I always require voxel-volumetric scaling instead of below
        # if fdk_weighting:
        #     scale = 1. / np.linalg.det([u, v, s])

        _det3x = lambda b, c: b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1]
        _det3y = lambda b, c: b[:, 0] * c[:, 2] - b[:, 2] * c[:, 0]
        _det3z = lambda b, c: b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]

        s_min_d = s - d
        numerator_u = cuda_float4(
            w=scale * xp.linalg.det(xp.asarray((s, v, d)).swapaxes(0, 1)),
            x=scale * _det3x(v, s_min_d),
            y=-scale * _det3y(v, s_min_d),
            z=scale * _det3z(v, s_min_d))
        numerator_v = cuda_float4(
            w=-scale * xp.linalg.det(xp.asarray((s, u, d)).swapaxes(0, 1)),
            x=-scale * _det3x(u, s_min_d),
            y=scale * _det3y(u, s_min_d),
            z=-scale * _det3z(u, s_min_d))
        denominator = cuda_float4(
            w=scale * xp.linalg.det(xp.asarray((u, v, s)).swapaxes(0, 1)),
            x=-scale * _det3x(u, v),
            y=scale * _det3y(u, v),
            z=-scale * _det3z(u, v))

        # if fdk_weighting:
        #     assert xp.allclose(denominator.w, 1.)

        return cp.asarray(
            numerator_u.to_list() +
            numerator_v.to_list() +
            denominator.to_list()).T
