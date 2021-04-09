import copy
from typing import Any, Callable, Sized

from tqdm import tqdm

from astrapy import process
from astrapy.data import *
from astrapy.geom3d import *
from astrapy.kernel import *
from astrapy.kernel import (_copy_to_symbol, _copy_to_texture, _cuda_float4,
                            _texture_shape)


def _normalize_geom(geometry: Geometry,
                    volume_extent_min: Sequence,
                    volume_extent_max: Sequence,
                    volume_voxel_size: Sequence,
                    volume_rotation: Sequence = (0., 0., 0.)):
    shift(geometry, -(np.array(volume_extent_min) + volume_extent_max) / 2)
    scale(geometry, np.array(volume_voxel_size))
    rotate_inplace(geometry, *volume_rotation)


class ConeProjection(Kernel):
    SLICES_PER_THREAD = 4
    COLS_PER_THREAD = 32
    ROWS_PER_BLOCK = 32

    def __init__(self, *args, **kwargs):
        super().__init__('cone_fp.cu', *args, **kwargs)

    def __call__(
        self,
        volume_texture: txt.TextureObject,
        volume_extent_min: Sequence,
        volume_extent_max: Sequence,
        geometries: list,
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

        # TODO(Adriaan): explicitly formalize that multiple detectors is
        #   problematic atm. Maybe consider more advanced geometries where
        #   the detector is shared.
        #   Note that multiple detectors might not be a problem, they
        #   just render different u's and v's, and if they factually have
        #   less rows and columns, the computation is silently doing
        #   something that is similar to supersampling.
        assert len(projections) == len(geometries)
        assert np.all([g.detector.rows == geometries[0].detector.rows
                       for g in geometries])
        assert np.all([g.detector.cols == geometries[0].detector.cols
                       for g in geometries])
        rows = geometries[0].detector.rows
        cols = geometries[0].detector.cols

        for proj, geom in zip(projections, geometries):
            if not isinstance(proj, cp.ndarray):
                raise TypeError("`projections` must be a CuPy ndarray.")
            if proj.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for "
                    f"dtype={self.SUPPORTED_DTYPES}.")
            if proj.shape != (geom.detector.rows, geom.detector.cols):
                raise ValueError("Projection shapes need to "
                                 "match detector (rows, cols).")

        if rays_per_pixel != 1:
            raise NotImplementedError(
                "Detector supersampling is currently not supported.")

        volume_shape = _texture_shape(volume_texture)
        if not has_isotropic_voxels(
            volume_shape, volume_extent_min, volume_extent_max):
            raise NotImplementedError(
                f"`{self.__class__.__name__}` is not tested with anisotropic "
                f"voxels yet.")

        vox_size = voxel_size(volume_shape, volume_extent_min,
                              volume_extent_max)

        proj_ptrs = [p.data.ptr for p in projections]
        assert len(proj_ptrs) == len(set(proj_ptrs)), (
            "All projection files need to reside in own memory.")

        norm_geoms = copy.deepcopy(geometries)
        for g in norm_geoms:
            _normalize_geom(g, volume_extent_min, volume_extent_max, vox_size,
                            volume_rotation)

        names = ['cone_fp<DIR_X>', 'cone_fp<DIR_Y>', 'cone_fp<DIR_Z>']
        module = self._compile(
            names=names,
            template_kwargs={'slices_per_thread': self.SLICES_PER_THREAD,
                    'cols_per_thread': self.COLS_PER_THREAD,
                    'nr_projs_global': len(norm_geoms)})
        funcs = [module.get_function(n) for n in names]
        self._upload_geometries(norm_geoms, module)

        output_scale = self._output_scale(vox_size)
        blocks_u = int(np.ceil(cols / self.COLS_PER_THREAD))
        blocks_v = int(np.ceil(rows / self.ROWS_PER_BLOCK))

        def _launch(start: int, stop: int, axis: int):
            # print(f"Launching {start}-{stop}, axis {axis}")
            # slice volume in the axis of the ray direction
            for i in range(0, volume_shape[axis], self.SLICES_PER_THREAD):
                funcs[axis](
                    (blocks_v, blocks_u, stop - start),
                    (self.ROWS_PER_BLOCK,),
                    (volume_texture,
                     cp.array(proj_ptrs),
                     i,  # start slice
                     start,  # projection
                     *volume_shape,
                     rows, cols,
                     *cp.float32(output_scale[axis])))

        def ray_direction_ax(g) -> int:
            d = np.abs(g.tube_position - g.detector_position)
            return np.argmax(d)  # noqa

        # Run over all angles, grouping them into groups of the same
        # orientation (roughly horizontal vs. roughly vertical).
        # Start a stream of grids for each such group.
        # TODO(Adriaan): sort and launch?
        start = 0
        prev_ax = ray_direction_ax(norm_geoms[start])
        for i, g in enumerate(norm_geoms):
            # if direction changed: launch kernel for this batch
            _ax = ray_direction_ax(g)
            if _ax != prev_ax:
                _launch(start, i, prev_ax)
                start = i
                prev_ax = _ax

        # launch kernel for remaining projections
        _launch(start, len(norm_geoms), prev_ax)

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
    def _upload_geometries(geometries: list, module: RawModule):
        """Transfer geometries to device as structure of arrays."""
        # TODO(Adriaan): maybe make a mapping between variables
        srcsX = np.array([g.tube_position[0] for g in geometries])
        srcsY = np.array([g.tube_position[1] for g in geometries])
        srcsZ = np.array([g.tube_position[2] for g in geometries])
        detsSX = np.array([g.detector_extent_min[0] for g in geometries])
        detsSY = np.array([g.detector_extent_min[1] for g in geometries])
        detsSZ = np.array([g.detector_extent_min[2] for g in geometries])
        dUX = np.array([g.u[0] * g.detector.pixel_width for g in geometries])
        dUY = np.array([g.u[1] * g.detector.pixel_width for g in geometries])
        dUZ = np.array([g.u[2] * g.detector.pixel_width for g in geometries])
        dVX = np.array([g.v[0] * g.detector.pixel_height for g in geometries])
        dVY = np.array([g.v[1] * g.detector.pixel_height for g in geometries])
        dVZ = np.array([g.v[2] * g.detector.pixel_height for g in geometries])
        _copy_to_symbol(module, 'detsSX', detsSX)
        _copy_to_symbol(module, 'detsSY', detsSY)
        _copy_to_symbol(module, 'detsSZ', detsSZ)
        _copy_to_symbol(module, 'srcsX', srcsX)
        _copy_to_symbol(module, 'srcsY', srcsY)
        _copy_to_symbol(module, 'srcsZ', srcsZ)
        _copy_to_symbol(module, 'detsUX', dUX)
        _copy_to_symbol(module, 'detsUY', dUY)
        _copy_to_symbol(module, 'detsUZ', dUZ)
        _copy_to_symbol(module, 'detsVX', dVX)
        _copy_to_symbol(module, 'detsVY', dVY)
        _copy_to_symbol(module, 'detsVZ', dVZ)


class ConeBackprojection(Kernel):
    # last dim is not number of threads, but volume slab thickness
    VOL_BLOCK = (16, 32, 6)
    LIMIT_PROJS_PER_BLOCK = 32

    def __init__(self):
        super().__init__('cone_bp.cu')

    def __call__(self,
                 projections_textures: Any,
                 geometries: list,
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
            for p, g in zip(projections_textures, geometries):
                if not _texture_shape(p) == (g.detector.rows, g.detector.cols):
                    raise ValueError("Projection texture resource needs to"
                                     " match the detector dimensions in the"
                                     " geometries.")

            compile_use_texture3D = False
            projections = cp.array([p.ptr for p in projections_textures])
        elif isinstance(projections_textures, txt.TextureObject):
            assert np.all([g.detector.rows == geometries[0].detector.rows
                           and g.detector.cols == geometries[0].detector.cols
                           for g in geometries])
            compile_use_texture3D = True
            projections = projections_textures
        else:
            raise ValueError("Please give in a list of 2D TextureObject, or "
                             "one 3D TextureObject.")

        vox_size = voxel_size(
            volume.shape, volume_extent_min, volume_extent_max)
        vox_volume = voxel_volume(
            volume.shape, volume_extent_min, volume_extent_max)

        module = self._compile(
            names=('cone_bp',),
            template_kwargs={'vol_block_x': self.VOL_BLOCK[0],
                    'vol_block_y': self.VOL_BLOCK[1],
                    'vol_block_z': self.VOL_BLOCK[2],
                    'nr_projs_block': self.LIMIT_PROJS_PER_BLOCK,
                    'nr_projs_global': len(geometries),
                    'texture3D': compile_use_texture3D}
        )
        cone_bp = module.get_function("cone_bp")

        params = []  # precomputed kernel parameters
        for g in copy.deepcopy(geometries):
            _normalize_geom(
                g, volume_extent_min, volume_extent_max,
                vox_size, volume_rotation)
            nU, nV, d = self._geom2params(g, vox_size, False)  # TODO
            # TODO(Adriaan): better to have SoA instead AoS?
            params.extend([*nU.to_list(), *nV.to_list(), *d.to_list()])
        _copy_to_symbol(module, 'params', np.array(params).astype(np.float))
        blocks = np.ceil(
            np.asarray(volume.shape) / self.VOL_BLOCK).astype(np.int)
        for start in range(0, len(geometries), self.LIMIT_PROJS_PER_BLOCK):
            cone_bp((blocks[0] * blocks[1], blocks[2]),
                    (self.VOL_BLOCK[0], self.VOL_BLOCK[1]),
                    (projections,
                     volume,
                     start,
                     len(geometries),
                     *volume.shape,
                     cp.float32(vox_volume)))

    @staticmethod
    def _geom2params(geom: Geometry,
                     voxel_size: Sequence,
                     fdk_weighting: bool = False):
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
        u = geom.u * geom.detector.pixel_width
        v = geom.v * geom.detector.pixel_height
        s = geom.tube_position
        d = geom.detector_extent_min

        # NB(ASTRA): for cross(u,v) we invert the volume scaling (for the voxel
        # size normalization) to get the proper dimensions for
        # the scaling of the adjoint
        cr = np.cross(u, v) * [voxel_size[1] * voxel_size[2],
                               voxel_size[0] * voxel_size[2],
                               voxel_size[0] * voxel_size[1]]
        assert np.linalg.det([u, v, s - d]) != 0.
        scale = np.sqrt(np.linalg.norm(cr)) / np.linalg.det([u, v, d - s])

        # TODO(Adriaan): it looks like my preweighting is different to ASTRA's
        #   and I always require voxel-volumetric scaling instead of below
        # if fdk_weighting:
        #     scale = 1. / np.linalg.det([u, v, s])

        _det3x = lambda b, c: b[1] * c[2] - b[2] * c[1]
        _det3y = lambda b, c: -(b[0] * c[2] - b[2] * c[0])
        _det3z = lambda b, c: b[0] * c[1] - b[1] * c[0]

        numerator_u = _cuda_float4(
            w=scale * np.linalg.det([s, v, d]),
            x=scale * _det3x(v, s - d),
            y=scale * _det3y(v, s - d),
            z=scale * _det3z(v, s - d))
        numerator_v = _cuda_float4(
            w=-scale * np.linalg.det([s, u, d]),
            x=-scale * _det3x(u, s - d),
            y=-scale * _det3y(u, s - d),
            z=-scale * _det3z(u, s - d))
        denominator = _cuda_float4(
            w=scale * np.linalg.det([u, v, s]),
            x=-scale * _det3x(u, v),
            y=-scale * _det3y(u, v),
            z=-scale * _det3z(u, v))

        if fdk_weighting:
            assert np.allclose(denominator.w, 1.)

        return numerator_u, numerator_v, denominator


def chunk_coneprojection(
    kernel: ConeProjection,
    volume,
    volume_extent_min,
    volume_extent_max,
    chunk_size: int,
    geometries: list,
    projections_cpu: np.ndarray,
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
    assert chunk_size > 0
    volume_texture = _copy_to_texture(volume)
    with cp.cuda.stream.Stream() as stream:
        for start_proj in tqdm(range(0, len(geometries), chunk_size),
            desc="Forward projecting"):
            next_proj = min(start_proj + chunk_size, len(geometries))
            sub_projs = projections_cpu[start_proj:next_proj]
            projs_gpu = [cp.zeros(p.shape, dtype=dtype) for p in sub_projs]
            stream.synchronize()

            kernel(volume_texture,
                   volume_extent_min,
                   volume_extent_max,
                   geometries[start_proj:next_proj],
                   projs_gpu,
                   **kwargs)
            stream.synchronize()

            for cpu, gpu in zip(sub_projs, projs_gpu):
                # TODO: check performance is improved with async/pinned memory
                cpu[:] = gpu.get()

            yield


def chunk_conebackprojection(
    kernel: ConeBackprojection,
    projections_cpu: np.ndarray,
    geometries: list,
    volume_shape: Sequence,
    volume_extent_min: Sequence,
    volume_extent_max: Sequence,
    chunk_size: int,
    dtype=cp.float32,
    filter: Any = None,
    preproc_fn: Callable = None,
    **kwargs):
    """
    Allocates GPU memory for only `chunk_size` projection images, then
    repeats the kernel call into the same GPU memory.
    """
    assert chunk_size > 0
    #  keeping the kernels write in the last dim
    volume = cp.zeros(tuple(volume_shape), dtype=dtype)
    with cp.cuda.stream.Stream() as stream:
        for start in tqdm(range(0, len(geometries), chunk_size),
                          desc="Backprojecting"):
            end = min(start + chunk_size, len(geometries))
            sub_geoms = geometries[start:end]
            sub_projs = projections_cpu[start:end]
            sub_projs_gpu = cp.asarray(sub_projs)

            if preproc_fn is not None:
                preproc_fn(sub_projs_gpu)
                stream.synchronize()

            process.preweight(sub_projs_gpu, sub_geoms)
            if filter is not None:
                process.filter(sub_projs_gpu)
                stream.synchronize()

            projs_txt = _copy_to_texture(sub_projs_gpu)
            stream.synchronize()

            kernel(
                projs_txt,
                sub_geoms,
                volume,
                volume_extent_min,
                volume_extent_max)
            stream.synchronize()

            yield volume
