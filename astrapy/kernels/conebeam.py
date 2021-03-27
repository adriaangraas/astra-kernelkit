import copy
from typing import Any, Sequence

from tqdm import tqdm

from astrapy import kernels
from astrapy.data import has_isotropic_voxels, voxel_size, voxel_volume
from astrapy.geom3d import AstraStatic3DGeometry, scale, shift
from astrapy.kernel import *
from astrapy.kernel import (
    _cuda_float4,
    _cupy_copy_to_constant,
    _cupy_copy_to_texture)


def _shift_and_scale(geometry: AstraStatic3DGeometry,
                     volume_extent_min: Sequence,
                     volume_extent_max: Sequence,
                     volume_voxel_size: Sequence):
    # shift geom to the center of the volume
    shift(geometry, -(np.array(volume_extent_min) + volume_extent_max) / 2)
    # compensating for anisotropic voxels
    scale(geometry, np.array(volume_voxel_size))


class ConeProjection(Kernel):
    BLOCK_SLICES = 4
    COLS_PER_BLOCK = 32
    ROWS_PER_BLOCK = 32
    LIMIT_BLOCK = 4

    def __init__(self, path: str = "../cuda/3d/cone_fp.cu"):
        super().__init__(path)

    def _compile(self, nr_projs_block: int, nr_projs_global: int):
        names = ['cone_fp<DIR_X>', 'cone_fp<DIR_Y>', 'cone_fp<DIR_Z>']
        module = self.load_module(
            name_expressions=names,
            columns_per_block=self.COLS_PER_BLOCK,
            rows_per_block=self.ROWS_PER_BLOCK,
            block_slices=self.BLOCK_SLICES,
            nr_projs_block=nr_projs_block,
            nr_projs_global=nr_projs_global)
        funcs = [module.get_function(name) for name in names]
        return module, funcs

    def __call__(
        self,
        volume_texture: txt.TextureObject,
        vol_shape: Sequence,
        vol_extent_min: Sequence,
        vol_extent_max: Sequence,
        geometries: list,
        projections: list,  # TODO: list[numpy.typing.ArrayLike]
        rays_per_pixel: int = 1
    ):
        """Forward projection with conebeam geometry.

        Remember: cuda don't do global memory allocation. `projections`
        need to be pre-allocated on the GPU or this will fail.

        All the projection angles are chopped up in batches of MAX_ANGLES.

        Then a loop is initiated over the angles in the batch. For every new
        angle the major direction is inferred.

        One kernel is launched on a grid of blocks with grid-dimensions:
            (nr-regions, nr-angle-batches)
        with
         - each regions size PIXELS_PER_BLOCK_U * PIXELS_PER_BLOCK_V
         - an angle batch size of MAX_ANGLES

        So each block
         - owns a small portion of the detector;
         - owns a consecutive sequence of projection angles.
        and runs a thread for each combination
            (pixel-u, projection-angle).

        Then that thread loops through a number on pixels in the row.
        """
        assert len(projections) == len(geometries)
        # TODO(Adriaan): explicitly formalize that multiple detectors is
        #   problematic atm. Maybe consider more advanced geometries where
        #   the detector is shared.
        #   Note that multiple detectors is technically not a problem, they
        #   just render different u's and v's, and if they factually have
        #   less rows and columns, the computation is silently doing
        #   something that is similar to supersampling
        assert np.all(
            g.detector.rows == geometries[0].detector.rows
            for g in geometries)
        assert np.all(
            g.detector.cols == geometries[0].detector.cols
            for g in geometries)
        rows = geometries[0].detector.rows
        cols = geometries[0].detector.cols

        for proj in projections:
            if not isinstance(proj, cp.ndarray):
                raise TypeError("`projections` must be a CuPy ndarray.")

            if proj.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for "
                    f"dtype={self.SUPPORTED_DTYPES}.")

        # TODO(Adriaan): check if texture object size corresponds to vol size?
        # if volume.data.dtype not in self.SUPPORTED_DTYPES:
        #     raise NotImplementedError(
        #         f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")

        vox_size = voxel_size(vol_shape,
                              vol_extent_min,
                              vol_extent_max)

        geometries = copy.deepcopy(geometries)
        [_shift_and_scale(g,
                          vol_extent_min,
                          vol_extent_max,
                          vox_size) for g in geometries]

        vol_shape = np.flip(vol_shape)  # TODO
        # TODO: possibly extents are off too

        proj_pointers = cp.array([p.data.ptr for p in projections])

        module, funcs = self._get_compilation(
            nr_projs_block=self.LIMIT_BLOCK,
            nr_projs_global=len(geometries))

        self._upload_geometries(geometries, module)
        output_scale = self._output_scale(vox_size)

        blocks_u = np.ceil(cols / self.COLS_PER_BLOCK).astype(np.int)
        blocks_v = np.ceil(rows / self.ROWS_PER_BLOCK).astype(np.int)

        def _launch(start: int, stop: int, axis: int):
            print(f"Launching {start}-{stop}, axis {axis}")
            blocks_projs = np.ceil(
                (stop - start) / self.LIMIT_BLOCK).astype(np.int)

            # TODO(ASTRA): check if we can't immediately destroy
            #              the stream after use
            # TODO(Adriaan):
            #   Now using `with stream.Stream()` but I should check
            #   if/how to run this loop with async streams (or
            #   are streams automatically async in CuPy?) in order to maximize
            #   performance. This could be a bottleneck atm, really
            #   need to check.
            with cp.cuda.stream.Stream():
                for i in range(0, vol_shape[axis], self.BLOCK_SLICES):
                    if rays_per_pixel != 1:
                        raise NotImplementedError(
                            "Detector supersampling is currently not supported.")

                    # TODO(Adriaan): why we are not using a 3D grid here?
                    funcs[axis](
                        (blocks_u * blocks_v, blocks_projs),
                        (self.COLS_PER_BLOCK, self.LIMIT_BLOCK),
                        (volume_texture,
                         proj_pointers,
                         i,  # startSlice
                         start,  # first projection
                         stop,  # end projection
                         *vol_shape,
                         cols,
                         rows,
                         cp.float32(output_scale[axis][0]),
                         cp.float32(output_scale[axis][1]),
                         cp.float32(output_scale[axis][2])))


        # Run over all angles, grouping them into groups of the same
        # orientation (roughly horizontal vs. roughly vertical).
        # Start a stream of grids for each such group.
        start = 0
        prev_ax = geometries[start].main_axis()
        for i in range(len(geometries)):
            # if direction changed: launch kernel for this batch
            _ax = geometries[i].main_axis()
            if _ax != prev_ax:
                _launch(start, i, prev_ax)
                prev_ax = _ax
                start = i

        # launch kernel for remaining projections
        _launch(start, len(geometries), prev_ax)
        cp.cuda.Device().synchronize()

    @staticmethod
    def _output_scale(voxel_size: np.ndarray) -> list:
        """Returns an output scale for each axis"""
        eps = .0001  # TODO(Adriaan): think about the precision of this factor
        if (abs(voxel_size[0] / voxel_size[1] - 1.0) > eps or
            abs(voxel_size[0] / voxel_size[2] - 1.0) > eps):
            output_scale_x = [(voxel_size[1] / voxel_size[0]) ** 2,
                              (voxel_size[2] / voxel_size[0]) ** 2,
                              voxel_size[0] ** 2 ]
            output_scale_y = [(voxel_size[0] / voxel_size[1]) ** 2,
                              (voxel_size[2] / voxel_size[1]) ** 2,
                              voxel_size[0] * voxel_size[1]]
            output_scale_z = [(voxel_size[0] / voxel_size[2]) ** 2,
                              (voxel_size[1] / voxel_size[2]) ** 2,
                              voxel_size[0] * voxel_size[2]]
            output_scale = [output_scale_x, output_scale_y, output_scale_z]
        else:  # cube is almost the same for all directions, thus * 3
            output_scale = [[1., 1., voxel_size[0]]] * 3

        return output_scale

    @staticmethod
    def _upload_geometries(geometries: list, module: RawModule):
        """Transfer geometries to device as structure of arrays."""
        # TODO(Adriaan): maybe make a mapping between variables
        srcsX = np.array([g.tube_position[0] for g in geometries])
        _cupy_copy_to_constant(module, 'srcsX', srcsX)
        srcsY = np.array([g.tube_position[1] for g in geometries])
        _cupy_copy_to_constant(module, 'srcsY', srcsY)
        srcsZ = np.array([g.tube_position[2] for g in geometries])
        _cupy_copy_to_constant(module, 'srcsZ', srcsZ)
        detsSX = np.array([g.detector_corner[0] for g in geometries])
        _cupy_copy_to_constant(module, 'detsSX', detsSX)
        detsSY = np.array([g.detector_corner[1] for g in geometries])
        _cupy_copy_to_constant(module, 'detsSY', detsSY)
        detsSZ = np.array([g.detector_corner[2] for g in geometries])
        _cupy_copy_to_constant(module, 'detsSZ', detsSZ)
        detsUX = np.array(
            [g.u[0] * g.detector.pixel_width for g in geometries])
        _cupy_copy_to_constant(module, 'detsUX', detsUX)
        detsUY = np.array(
            [g.u[1] * g.detector.pixel_width for g in geometries])
        _cupy_copy_to_constant(module, 'detsUY', detsUY)
        detsUZ = np.array(
            [g.u[2] * g.detector.pixel_width for g in geometries])
        _cupy_copy_to_constant(module, 'detsUZ', detsUZ)
        detsVX = np.array(
            [g.v[0] * g.detector.pixel_height for g in geometries])
        _cupy_copy_to_constant(module, 'detsVX', detsVX)
        detsVY = np.array(
            [g.v[1] * g.detector.pixel_height for g in geometries])
        _cupy_copy_to_constant(module, 'detsVY', detsVY)
        detsVZ = np.array(
            [g.v[2] * g.detector.pixel_height for g in geometries])
        _cupy_copy_to_constant(module, 'detsVZ', detsVZ)


class ConeBackprojection(Kernel):
    # last dim is not number of threads, but volume slab thickness
    VOL_BLOCK = (16, 32, 6)
    LIMIT_BLOCK = 32

    def __init__(self, path: str = "../cuda/3d/cone_bp.cu",
                 *args, **kwargs):
        super().__init__(path, *args, **kwargs)

    def _compile(self, nr_projs_block: int, nr_projs_global: int):
        module = self.load_module(
            name_expressions=('cone_bp<true>', 'cone_bp<false>'),
            vol_block_x=self.VOL_BLOCK[0],
            vol_block_y=self.VOL_BLOCK[1],
            vol_block_z=self.VOL_BLOCK[2],
            nr_projs_block=nr_projs_block,
            nr_projs_global=nr_projs_global)
        funcs = {True: module.get_function("cone_bp<true>"),
                 False: module.get_function("cone_bp<false>")}
        return module, funcs

    def __call__(self,
                 projections_texture: txt.TextureObject,
                 geometries: list,
                 volume: cp.ndarray,
                 volume_shape: Sequence,
                 vol_extent_min: Sequence,
                 vol_extent_max: Sequence,
                 fdk_weighting: bool = False):
        """Backprojection with conebeam geometry."""

        # TODO: detectors could differ, this would screw up the scaling
        assert np.all(
            [g.detector.pixel_width == geometries[0].detector.rows] for g in
            geometries)

        if isinstance(volume, cp.ndarray):
            if volume.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for "
                    f"dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`volume` must be a CuPy ndarray.")

        # TODO: test, and also apply to other cuda
        assert volume.flags.c_contiguous is True, \
            f"`{self.__class__.__name__}` is not tested without " \
            f"C-contiguous data."

        if not has_isotropic_voxels(volume_shape, vol_extent_min,
                                    vol_extent_max):
            raise NotImplementedError(
                f"`{self.__class__.__name__}` is not tested with anisotropic "
                f"voxels yet.")

        volume_shape = np.array(volume_shape)
        vol_extent_min = np.array(vol_extent_min)
        vol_extent_max = np.array(vol_extent_max)
        volume_shape = np.flip(volume_shape)  # TODO
        vol_extent_min = np.flip(vol_extent_min)  # TODO
        vol_extent_max = np.flip(vol_extent_max)  # TODO

        vox_size = voxel_size(volume_shape, vol_extent_min,
                              vol_extent_max)
        vox_volume = voxel_volume(volume_shape, vol_extent_min,
                                  vol_extent_max)

        geometries = copy.deepcopy(geometries)
        for g in geometries:
            _shift_and_scale(g, vol_extent_min, vol_extent_max, vox_size)

        output_scale = 1.  # params.fOutputScale seems to be ~1. by default
        if fdk_weighting:  # ASTRA: NB: assuming cube voxels here
            output_scale /= vox_size[0]  # TODO
            if not np.all(vox_size == vox_size[0]):
                raise NotImplementedError('Anisotropic voxels are not tested.')
        else:
            output_scale *= vox_volume

        blocks = np.ceil(volume_shape / self.VOL_BLOCK).astype(np.int)

        # precompute kernel parameters
        geom_params = [ConeBackprojection._geom2params(
            geom, fdk_weighting, vox_size) for geom in geometries]

        # TODO(Adriaan): was it not better to have all params grouped?
        #  for performance, AoS vs SoA
        params_list = [[*nU.to_list(), *nV.to_list(), *d.to_list()]
                       for (nU, nV, d) in geom_params]
        params = np.array(params_list).flatten().astype(np.float)
        assert not np.isnan(params).any()

        module, funcs = self._get_compilation(
            nr_projs_block=self.LIMIT_BLOCK,
            nr_projs_global=len(geometries))
        _cupy_copy_to_constant(module, 'params', params)

        with cp.cuda.stream.Stream():
            for start in range(0, len(geometries), self.LIMIT_BLOCK):
                funcs[fdk_weighting](
                    (blocks[0] * blocks[1], blocks[2]),
                    (self.VOL_BLOCK[0], self.VOL_BLOCK[1]),
                    (projections_texture,
                     volume,
                     start,
                     len(geometries),
                     *volume_shape,
                     cp.float32(output_scale)))

        cp.cuda.Device().synchronize()

        output_scale /= float(vox_volume)
        output_scale *= float(geometries[0].detector.pixel_volume)

        # TODO(Adriaan): ODL would be calling
        #    # Fix scaling to weight by pixel/voxel size
        #    out *= astra_cuda_bp_scaling_factor(
        #        self.proj_space, self.vol_space, self.geometry
        #    )
        #  here, adding an additional factor.

    @staticmethod
    def _geom2params(geom: AstraStatic3DGeometry,
                     fdk_weighting: bool,
                     voxel_size: Sequence):
        """TODO(Adriaan): vectorize, using Numpy/CuPy

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
        u = geom.u * geom.detector.pixel_width
        v = geom.v * geom.detector.pixel_height
        s = geom.tube_position
        d = geom.detector_corner

        if not fdk_weighting:
            cr = np.cross(u, v)
            # NB(ASTRA): for cross(u,v) we invert the volume scaling (for the voxel
            # size normalization) to get the proper dimensions for
            # the scaling of the adjoint
            cr *= [voxel_size[1] * voxel_size[2],
                   voxel_size[0] * voxel_size[2],
                   voxel_size[0] * voxel_size[1]]
            scale = np.sqrt(np.linalg.norm(cr)) / np.linalg.det([u, v, s - d])
        else:
            scale = 1. / np.linalg.det([u, v, s])

        _det3x = lambda b, c: b[1] * c[2] - b[2] * c[1]
        _det3y = lambda b, c: -(b[0] * c[2] - b[2] * c[0])
        _det3z = lambda b, c: b[0] * c[1] - b[1] * c[0]

        numU = _cuda_float4(w=scale * np.linalg.det([s, v, d]),
                            x=scale * _det3x(v, s - d),
                            y=scale * _det3y(v, s - d),
                            z=scale * _det3z(v, s - d))
        numV = _cuda_float4(w=-scale * np.linalg.det([s, u, d]),
                            x=-scale * _det3x(u, s - d),
                            y=-scale * _det3y(u, s - d),
                            z=-scale * _det3z(u, s - d))
        den = _cuda_float4(w=scale * np.linalg.det([u, v, s]),  # = 1.0 for FDK
                           x=-scale * _det3x(u, v),
                           y=-scale * _det3y(u, v),
                           z=-scale * _det3z(u, v))
        return numU, numV, den


def chunk_coneprojection(
    kernel: ConeProjection,
    volume,
    volume_extent_min,
    volume_extent_max,
    chunk_size: int,
    geometries,
    projections_cpu,
    dtype=cp.float32,
    **kwargs):
    """
    Allocates GPU memory for only `chunk_size` projection images, then
    repeats the kernel call into the same GPU memory.

    :param kernel:
    :param chunk_size:
    :param geometries:
    :param projections_cpu:
    :param kwargs:
    """
    if not issubclass(type(kernel), Kernel):
        raise ValueError("Only `Kernel` subtypes are allowed.")

    assert chunk_size > 0

    volume_texture = _cupy_copy_to_texture(volume)

    for start_proj in range(0, len(geometries), chunk_size):
        next_proj = min(start_proj + chunk_size, len(geometries))
        sub_geoms = geometries[start_proj:next_proj]
        sub_projs = projections_cpu[start_proj:next_proj]

        # upload chunk of projections
        projs_gpu = []
        for p in sub_projs:
            projs_gpu.append(cp.zeros(p.shape, dtype=dtype))

        # launch kernel for these projs
        kernel(volume_texture,
               volume.shape,
               volume_extent_min,
               volume_extent_max,
               sub_geoms,
               projs_gpu,
               **kwargs)

        # copy projs back to CPU
        for cpu, gpu in zip(sub_projs, projs_gpu):
            cpu[:] = gpu.get()

        yield


def chunk_conebackprojection(
    kernel: ConeBackprojection,
    projections_cpu,
    geometries,
    volume_shape,
    volume_extent_min,
    volume_extent_max,
    chunk_size: int,
    dtype=cp.float32,
    filter: Any = None,
    fpreproc=None,
    **kwargs):
    """
    Allocates GPU memory for only `chunk_size` projection images, then
    repeats the kernel call into the same GPU memory.
    """
    if not issubclass(type(kernel), Kernel):
        raise ValueError("Only `Kernel` subtypes are allowed.")

    assert chunk_size > 0

    volume = cp.zeros(volume_shape, dtype=dtype)

    for start in tqdm(range(0, len(geometries), chunk_size)):
        end = min(start + chunk_size, len(geometries))
        sub_geoms = geometries[start:end]
        sub_projs = projections_cpu[start:end]

        if fpreproc is not None:
            sub_projs_gpu = cp.asarray(sub_projs)
            fpreproc(sub_projs_gpu)
        else:  # prevent copying to GPU if not necessary
            sub_projs_gpu = False

        if filter is not None:
            filter_kernel = kernels.Filter()
            if sub_projs_gpu is False:
                sub_projs_gpu = cp.asarray(sub_projs)

            filter_kernel(sub_projs_gpu, sub_geoms, filter,
                          volume_extent_min, volume_extent_max,
                          voxel_size(volume_shape,
                                     volume_extent_min,
                                     volume_extent_max))

        if sub_projs_gpu is not False:
            # if filtering, copy device-to-device
            proj_texture = _cupy_copy_to_texture(sub_projs_gpu)
        else:
            # if not filtering, copy directly host-to-device
            # TODO(Adriaan) create GPU textureObject without the
            #   Numpy `asarray` (which might make a copy?), by uploading
            #   the invidial projections directly to parts of texture memory?
            proj_texture = _cupy_copy_to_texture(np.asarray(sub_projs))

        # launch kernel for these projs
        kernel(
            proj_texture,
            sub_geoms,
            volume,
            volume_shape,
            volume_extent_min,
            volume_extent_max,
            fdk_weighting=filter is not None,
            **kwargs)

        yield volume