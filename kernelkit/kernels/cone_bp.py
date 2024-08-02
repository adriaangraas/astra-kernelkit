import copy
from enum import Enum
from importlib import resources
from typing import Any, Sequence

import cupy as cp
from cupy.cuda.texture import TextureObject
import numpy as np

from kernelkit import KERNELKIT_CUDA_SOURCES
from kernelkit.geom import normalize_
from kernelkit.geom.proj import ProjectionGeometrySequence
from kernelkit.geom.vol import VolumeGeometry
from kernelkit.kernel import (
    KernelMemoryUnpreparedException,
    KernelNotCompiledException,
    copy_to_symbol,
    cuda_float4,
    BaseKernel,
)


class VoxelDrivenConeBP(BaseKernel):
    """Voxel-driven conebeam backprojection kernel."""

    FLOAT_DTYPE = cp.float32
    SUPPORTED_DTYPES = [cp.float32]

    class InterpolationMethod(Enum):
        """Texture fetching approach.

        Notes
        -----
        The fetching approach (i.e., the texture fetching function to call in
        the kernel) does not correspond one-on-one to texture objects.
        """

        Tex3D = "tex3D"  # for single texture object from 3D CUDA Array
        Tex2DLayered = "tex2DLayered"  # yes, for layered *3D* CUDA array
        Surf2DLayered = "surf2DLayered"
        Tex2D = "tex2D"  # List of pitch2D, or list of 2D CUDA arrays

    voxels_per_block = (6, 32, 16)  # ASTRA Toolbox default
    projs_per_block = 32  # ASTRA Toolbox default

    def __init__(
        self,
        voxels_per_block: tuple = None,
        projs_per_block: int = None,
        volume_axes: tuple = (0, 1, 2),
        projection_axes: tuple = (0, 1, 2),
    ):
        """Conebeam backprojection kernel.

        Parameters
        ----------
        voxels_per_block : tuple, optional
            Number of voxels computed in one thread block, corresponding to
            the reversed `volume.shape`, not taking into account the mapping to
            world coordinates via `volume_axes`.
            I.e.: voxels_per_block[2] describes CUDA blockDim.x and
            voxels_per_block[0] describes how many slices `i` of
            `volume[i, ...]` are processed in one thread.
            If your input `volume` is (8, 16, 32) voxels, maybe a good setting
            would be (32, 16, 8), regardless to which world axes (x, y, z) the
            8, 16, and 32 voxels belong. Note
            `voxels_per_block[0] * voxels_per_block[1]`  is constrained to
            the maximum number of CUDA threads, typically 1024.
        projs_per_block : int, optional
            Maximum number of projections processed in one thread block.
            If `None` defaults to `LIMIT_PROJS_PER_BLOCK`.
        volume_axes : tuple, optional
            Axes of the backprojection volume. Defaults to `(0, 1, 2)`.
            The axes are in the order x, y, z. The first axis is the source-
            detector axis, the second is the horizontal orthogonal axis and
            the third is the vertical axis. ASTRA Toolbox uses (2, 1, 0).
        projection_axes : tuple, optional
            Axes of the projections to be backprojected. Defaults to
            `(0, 1, 2)`. The axes are in the order angle, row, column.

        See Also
        --------
        kernelkit.kernel.VoxelDrivenConeFP : Conebeam forward projection kernel
        kernelkit.kernel.BaseKernel : Base class for CUDA kernels
        """
        cuda_source = (
            resources
            .files(KERNELKIT_CUDA_SOURCES)
            .joinpath("cone_bp.cu")
            .read_text()
        )
        super().__init__(cuda_source)

        if voxels_per_block is not None and len(voxels_per_block) != 3:
            raise ValueError("`voxels_per_block` must have 3 elements.")
        self._vox_block = (
            voxels_per_block if voxels_per_block is not None else self.voxels_per_block
        )
        self._projs_block = (
            projs_per_block if projs_per_block is not None else self.projs_per_block
        )
        self._vol_axs = volume_axes
        self._proj_axs = projection_axes
        self._params_are_set = False

    @property
    def volume_axes(self):
        return self._vol_axs

    @property
    def projection_axes(self):
        return self._proj_axs

    def compile(
        self,
        max_projs: int = 1024,
        texture: InterpolationMethod | None | str = InterpolationMethod.Tex3D,
    ) -> cp.RawModule:
        """Compile the kernel.

        Parameters
        ----------
        max_projs : int, optional
            Maximum number of projections to be processed in one kernel.
            If `None` defaults to the ASTRA Toolbox default of 1024. The
            value is used to compile a fixed amount of constant memory into
            the kernel that store the geometry parameters. This means that
            the kernel can be launched with different numbers N_\theta of
            projections, but N_\theta must be smaller or equal to `max_projs`.
        texture : str, optional
            Type of texture to use. Defaults to `3D`. Can be one of the types
            specified in `InterpolationMethod`.
        """
        if texture is not None:
            if texture not in self.InterpolationMethod:
                raise ValueError(f"`texture` must be one of {self.InterpolationMethod}"
                                 f" or `None`.")

            if texture.value == self.InterpolationMethod.Tex2D.value:
                if self._proj_axs[0] != 0:
                    raise ValueError(
                        "If an array of texture pointers is given, the first "
                        "`projection_axes` axis must be angles, i.e., must be 0."
                    )
            elif texture.value in (
                self.InterpolationMethod.Tex2DLayered.value,
                # self.InterpolationMethod.Tex3D.value,
            ):
                if self._proj_axs[0] != 0:
                    raise ValueError(
                        "If a 3D texture is given, the first `projection_axes` "
                        "axis must be angles, i.e., must be 0."
                    )

        self._compile(
            name_expressions=("cone_bp",),
            template_kwargs={
                "voxels_per_block": self._vox_block,
                "nr_projs_block": self._projs_block,
                "nr_projs_global": max_projs,
                "texture": texture.value if texture is not None else texture,
                "volume_axes": self._vol_axs,
                "projection_axes": self._proj_axs,
            },
        )
        self._module.compile()

    def set_params(self, params: cp.ndarray):
        """Copy parameters to constant memory of the kernel.

        Parameters
        ----------
        params : cupy.ndarray
            Array of parameters for each projection. Can be obtained
            with `VoxelDrivenConeBP.geoms2params()`.
        """
        if not self.is_compiled:
            raise KernelNotCompiledException(
                f"Kernel {self.__class__.__name__} must be compiled before "
                f"uploading parameters to the GPU.")

        if len(params) // 12 > self._compiled_template_kwargs["nr_projs_global"]:
            raise ValueError(
                f"Number of projections, {len(params) // 12}, exceeds the "
                "the maximum of the compiled kernel, namely "
                f"{self._compiled_template_kwargs['nr_projs_global']}. "
                "Please recompile the kernel with a higher `max_projs`."
            )
        copy_to_symbol(self._module, "params", params)
        self._params_are_set = True

    def __call__(self, textures: Any, volume: Any, volume_geometry: VolumeGeometry):
        """XrayBackprojection with conebeam geometry.

        Parameters
        ----------
        textures : list[TextureObject] or TextureObject
            If `TextureObject` the kernel compiles with `texture3D` spec,
            and the kernel expects a 3D contiguous texture block and uses
            `tex3D()` to access it.
            If `list[TextureObject]` the kernel compiles to expect an array
            of pointers to 2D texture objects (one for each projection), and
            accesses the textures by a list of pointers.
        volume : cp.ndarray
            The volume to backproject into.
        volume_geometry : VolumeGeometry
            The volume geometry.

        Notes
        -----
         - A kernel must always be compiled via `compile()` before the first
        call.
         - The __call__ does not check if the dimensions of the textures
           match the geometry, for performance reasons.
        """
        if isinstance(volume, cp.ndarray):
            if volume.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Volume has a dtype '{volume.dtype}'. However, currently "
                    f"there is only support for dtype={self.SUPPORTED_DTYPES}."
                )
        else:
            raise TypeError("`volume` must be a CuPy ndarray.")
        assert (
            volume.flags.c_contiguous is True
        ), f"`{self.__class__.__name__}` is not tested without C-contiguous data."

        if not volume_geometry.has_isotropic_voxels():
            raise NotImplementedError(
                f"`{self.__class__.__name__}` is not tested with anisotropic "
                "voxels yet."
            )

        if not self.is_compiled:
            raise KernelNotCompiledException(
                "Please compile the kernel with `compile()`."
            )

        if isinstance(textures, cp.ndarray):
            if not textures.dtype == cp.int64:
                raise ValueError(
                    f"Please give in an array of int64 pointers to "
                    f"TextureObject. The given dtype is {textures.dtype}."
                )
            assert (
                self._compiled_template_kwargs["texture"]
                in (self.InterpolationMethod.Tex2D.value, None)
            ), (
                "Kernel was compiled with texture type "
                f"{self._compiled_template_kwargs['texture']}, "
                f"but given '{self.InterpolationMethod.Tex2D.value}'."
            )
            nr_projs = len(textures)
        elif isinstance(textures, TextureObject):
            assert textures.ResDesc.cuArr is not None, (
                "A single texture object for all the projections has been "
                "passed, but it does not contain a CUDA array attached."
            )
            if textures.ResDesc.cuArr.flags % 2 == 1:
                assert (
                    self._compiled_template_kwargs["texture"]
                    == self.InterpolationMethod.Tex2DLayered.value
                ), (
                    "Kernel was compiled with texture type "
                    f"{self._compiled_template_kwargs['texture']}, "
                    f"but given '{self.InterpolationMethod.Tex2DLayered.value}'."
                )
            else:
                if (
                    self._compiled_template_kwargs["texture"]
                    != self.InterpolationMethod.Tex3D.value
                ):
                    raise ValueError(
                        "Kernel was compiled with texture type "
                        f"{self._compiled_template_kwargs['texture']}, "
                        f"but given '{self.InterpolationMethod.Tex3D.value}'."
                    )
            cuArr = textures.ResDesc.cuArr
            dims = cuArr.depth, cuArr.height, cuArr.width
            nr_projs = dims[self._proj_axs[0]]
        else:
            raise ValueError(
                f"Please give in an array of pointers to "
                f"TextureObject, or one 3D TextureObject. Given is a "
                f"'{type(textures).__name__}'."
            )

        if not self._params_are_set:
            raise KernelMemoryUnpreparedException(
                "Please set the parameters with `set_params()`."
            )

        cone_bp = self._module.get_function("cone_bp")
        vox_vol = cp.float32(volume_geometry.voxel_volume())
        blocks = np.ceil(np.asarray(volume.shape) / self._vox_block).astype(np.int32)

        for start in range(0, nr_projs, self._projs_block):
            cone_bp(
                (blocks[2] * blocks[1], blocks[0]),
                (self._vox_block[2], self._vox_block[1]),
                (textures, volume, start, nr_projs, *volume.shape, vox_vol),
            )

    @staticmethod
    def geoms2params(
        projection_geometry, volume_geometry: VolumeGeometry, with_adjoint_scaling=True
    ):
        """Compute kernel parameters

        Parameters
        ----------
        projection_geometry : ProjectionGeometry or list[ProjectionGeometry]
            The projection geometries.
        volume_geometry : VolumeGeometry
            The volume geometry.

        Notes
        -----
        The following is taken from ASTRA Toolbox. We need three things
        in the kernel:
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
        if isinstance(projection_geometry, ProjectionGeometrySequence):
            geom_seq = copy.deepcopy(projection_geometry)
        else:
            # list, tuple, or ndarray
            geom_seq = ProjectionGeometrySequence.fromList(projection_geometry)

        xp = geom_seq.xp
        normalize_(geom_seq, volume_geometry)
        vox_size = volume_geometry.voxel_size
        u = geom_seq.u * geom_seq.detector.pixel_width[..., xp.newaxis]
        v = geom_seq.v * geom_seq.detector.pixel_height[..., xp.newaxis]
        s = geom_seq.source_position
        d = geom_seq.detector_extent_min

        # NB(ASTRA): for cross(u,v) we invert the volume factor (for the voxel
        # size normalization) to get the proper dimensions for
        # the factor of the adjoint
        # TODO(Adriaan): I think the concept of scaling the backprojector
        #   should be moved out of the kernel. It looks like it is
        #   complicating the code here tremendously, and I'm not if
        #   it is necessary to do it together with backprojection.
        cr = xp.cross(u, v)  # maintain f32
        cr *= xp.array(
            [
                vox_size[1] * vox_size[2],
                vox_size[0] * vox_size[2],
                vox_size[0] * vox_size[1],
            ]
        )

        scale = xp.sqrt(xp.linalg.norm(cr, axis=1))

        if with_adjoint_scaling:
            denom = xp.linalg.det(xp.asarray((u, v, d - s)).swapaxes(0, 1))
            # if denom == 0.0:
            #     raise Exception(
            #         "Cannot accurately determine a scaling factor for "
            #         "the voxel-driven kernel as an adjoint, since "
            #         "det(u, v, source - det) = 0.")
            scale /= denom
        else:
            scale = 1.0

        # TODO(Adriaan): it looks like my preweighting is different to ASTRA's
        #   and I always require voxel-volumetric factor instead of below
        # if fdk_weighting:
        #     scale = 1. / xp.linalg.det(xp.asarray([u, v, s]).swapaxes(0, 1))
        #     scale = 1. / np.linalg.det([u, v, s])

        def _det3x(b, c):
            return b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1]

        def _det3y(b, c):
            return b[:, 0] * c[:, 2] - b[:, 2] * c[:, 0]

        def _det3z(b, c):
            return b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]

        s_min_d = s - d
        numerator_u = cuda_float4(
            w=scale * xp.linalg.det(xp.asarray((s, v, d)).swapaxes(0, 1)),
            x=scale * _det3x(v, s_min_d),
            y=-scale * _det3y(v, s_min_d),
            z=scale * _det3z(v, s_min_d),
        )
        numerator_v = cuda_float4(
            w=-scale * xp.linalg.det(xp.asarray((s, u, d)).swapaxes(0, 1)),
            x=-scale * _det3x(u, s_min_d),
            y=scale * _det3y(u, s_min_d),
            z=-scale * _det3z(u, s_min_d),
        )
        denominator = cuda_float4(
            w=scale * xp.linalg.det(xp.asarray((u, v, s)).swapaxes(0, 1)),
            x=-scale * _det3x(u, v),
            y=scale * _det3y(u, v),
            z=-scale * _det3z(u, v),
        )
        return cp.asarray(
            numerator_u.to_list() + numerator_v.to_list() + denominator.to_list()
        ).T.flatten()
