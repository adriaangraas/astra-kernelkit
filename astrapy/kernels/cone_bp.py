import copy
from typing import Any, Sequence
import cupy as cp
from cupy.cuda.texture import TextureObject
import numpy as np

from astrapy.geom import normalize_
from astrapy.geom.proj import GeometrySequence
from astrapy.geom.vol import VolumeGeometry
from astrapy.kernel import (copy_to_symbol, cuda_float4, Kernel)


class ConeBackprojection(Kernel):
    """Conebeam backprojection kernel."""
    VOXELS_PER_BLOCK = (16, 32, 6)
    LIMIT_PROJS_PER_BLOCK = 32
    MAX_PROJS = 1024
    NUM_STREAMS = 2

    def __init__(self,
                 max_projs: int = None,
                 voxels_per_block: tuple = None,
                 limit_projs_per_block: int = None):
        """Conebeam backprojection kernel.

        Parameters
        ----------
        max_projs : int, optional
            Maximum number of projections to be processed in one kernel.
            If `None` defaults to `MAX_PROJS`.
        voxels_per_block : tuple, optional
            Number of voxels computed in one thread block.
            If `None` defaults to `VOXELS_PER_BLOCK`.
        limit_projs_per_block : int, optional
            Maximum number of projections processed in one thread block.
            If `None` defaults to `LIMIT_PROJS_PER_BLOCK`.
        """
        super().__init__('cone_bp.cu')
        self._min_limit_projs = (max_projs if max_projs is not None
                                 else self.MAX_PROJS)
        self._vox_block = (voxels_per_block if voxels_per_block is not None
                           else self.VOXELS_PER_BLOCK)
        self._limit_projs_per_block = (
            limit_projs_per_block if limit_projs_per_block is not None
            else self.LIMIT_PROJS_PER_BLOCK)

    def compile(self, nr_projs: int = None,
                use_texture_3d: bool = True) -> cp.RawModule:
        """Compile the kernel.

        Parameters
        ----------
        nr_projs : int, optional
            Number of projections to be processed in one kernel.
            If `None` defaults to `self._min_limit_projs`.
        use_texture_3d : bool, optional
            If `True` compiles the kernel to use 3D texture memory.
            If `False` compiles the kernel to use 2D texture memory.
            Defaults to `True`.
        """
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
                             'texture3D': use_texture_3d})

    def __call__(self,
                 projections_textures: Any,
                 params: Sequence,
                 volume: cp.ndarray,
                 volume_geometry: VolumeGeometry):
        """Backprojection with conebeam geometry.

        Parameters
        ----------
        projections_textures : list[TextureObject] or TextureObject
            If `TextureObject` the kernel compiles with `texture3D` spec,
            and the kernel expects a 3D contiguous texture block and uses
            `tex3D()` to access it.
            If `list[TextureObject]` the kernel compiles to expect an array
            of pointers to 2D texture objects (one for each projection), and
            accesses the textures by a list of pointers.
        params : Sequence
            Sequence of parameters for each projection.
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
        if not volume_geometry.has_isotropic_voxels():
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

        module = self.compile(len(params), compile_use_texture3D)
        cone_bp = module.get_function("cone_bp")
        copy_to_symbol(module, 'params', params.flatten())

        # TODO(Adriaan): better to have SoA instead AoS?
        blocks = np.ceil(np.asarray(volume.shape) / self._vox_block).astype(
            np.int32)
        streams = [cp.cuda.stream.Stream(non_blocking=True)
                   for _ in range(self.NUM_STREAMS)]
        for i, start in enumerate(
            range(0, len(params), self._limit_projs_per_block)):
            with streams[i % len(streams)]:
                cone_bp((blocks[0] * blocks[1], blocks[2]),  # grid
                        (self._vox_block[0], self._vox_block[1]),  # threads
                        (projections,
                         volume,
                         start,
                         len(params),
                         *volume.shape,
                         cp.float32(volume_geometry.voxel_volume)))
        [s.synchronize() for s in streams]

    @staticmethod
    def geoms2params(projection_geometry, volume_geometry: VolumeGeometry):
        """Compute kernel parameters

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
        if isinstance(projection_geometry, list):
            geom_seq = GeometrySequence.fromList(projection_geometry)
        else:
            geom_seq = copy.deepcopy(projection_geometry)

        xp = geom_seq.xp
        normalize_(geom_seq, volume_geometry)
        vox_size = volume_geometry.voxel_size
        u = geom_seq.u * geom_seq.detector.pixel_width[..., xp.newaxis]
        v = geom_seq.v * geom_seq.detector.pixel_height[..., xp.newaxis]
        s = geom_seq.source_position
        d = geom_seq.detector_extent_min

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
        return cp.asarray(
            numerator_u.to_list() +
            numerator_v.to_list() +
            denominator.to_list()).T
