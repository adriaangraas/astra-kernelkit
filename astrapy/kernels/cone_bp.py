from math import ceil
from typing import Any, Sequence, Sized
from cupy.cuda.texture import TextureObject
from astrapy.data import *
from astrapy.geom import *
from astrapy.kernel import (copy_to_symbol, cuda_float4, texture_shape,
                            Kernel)


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
        s = geometries.source_position
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
