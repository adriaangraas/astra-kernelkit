import copy
import warnings
from enum import Enum
from typing import Any, Sequence

import cupy as cp
from cupy.cuda.texture import TextureObject
import numpy as np

from kernelkit.geom import normalize_
from kernelkit.geom.proj import GeometrySequence
from kernelkit.geom.vol import VolumeGeometry
from kernelkit.kernel import (KernelMemoryUnpreparedException,
                              KernelNotCompiledException, copy_to_symbol,
                              cuda_float4, BaseKernel)


class VoxelDrivenFanBP(BaseKernel):
    """Voxel-driven conebeam backprojection kernel."""

    class TextureFetching(Enum):
        """Texture fetching approach.

        Notes
        -----
        The fetching approach (i.e., the texture fetching function to call in
        the kernel) does not correspond one-on-one to texture objects.
        """
        Tex2D = '2D'

    voxels_per_block = (16, 32)  # ASTRA Toolbox default
    projs_per_block = 16  # ASTRA Toolbox default

    def __init__(self,
                 voxels_per_block: tuple = None,
                 projs_per_block: int = None,
                 volume_axes: tuple = (0, 1, 2),
                 projection_axes: tuple = (0, 1, 2)):
        """Conebeam backprojection kernel.

        Parameters
        ----------
        voxels_per_block : tuple, optional
            Number of voxels computed in one thread block.
            If `None` defaults to `VOXELS_PER_BLOCK`.
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
        super().__init__('fan_bp.cu')
        if voxels_per_block is not None and len(voxels_per_block) != 3:
            raise ValueError('`voxels_per_block` must have 3 elements.')
        self._vox_block = (voxels_per_block if voxels_per_block is not None
                           else self.voxels_per_block)
        self._projs_block = (projs_per_block if projs_per_block is not None
                             else self.projs_per_block)
        self._vol_axs = volume_axes
        self._proj_axs = projection_axes
        self._params_are_set = False

    @property
    def volume_axes(self):
        return self._vol_axs

    @property
    def projection_axes(self):
        return self._proj_axs

    def compile(self,
                max_projs: int = 2560,
                texture: TextureFetching | str = TextureFetching.Tex2D) -> cp.RawModule:
        """Compile the kernel.

        Parameters
        ----------
        max_projs : int, optional
            Maximum number of projections to be processed in one kernel.
            If `None` defaults to the ASTRA Toolbox default of 1024. The
            value is used to compile a fixed amount of constant memory into
            the kernel that store the geometry parameters.
        texture : str, optional
            Type of texture to use. Defaults to `3D`. Can be one of the types
            specified in `TextureFetching`.
        """
        if texture not in self.TextureFetching:
            raise ValueError(
                f"`texture` must be one of {self.TextureFetching}.")

        if texture.value == self.TextureFetching.Tex2D.value:
            if self._proj_axs[0] != 0:
                raise ValueError("If an array of texture pointers is given, "
                                 "the first `projection_axes` axis must be "
                                 "angles, i.e., must be 0.")

        self._compile(
            name_expressions=('fan_bp',),
            template_kwargs={'nr_vxls_block_x': self._vox_block[0],
                             'nr_vxls_block_y': self._vox_block[1],
                             'nr_projs_block': self._projs_block,
                             'nr_projs_global': max_projs,
                             'texture': texture.value,
                             'volume_axes': self._vol_axs,
                             'projection_axes': self._proj_axs})
        self._module.compile()

    def set_params(self, params: Sequence):
        """Copy parameters to constant memory of the kernel.

        Parameters
        ----------
        params : Sequence
            Sequence of parameters for each projection. Can be obtained
            with `VoxelDrivenConeBP.geoms2params()`.
        """
        if len(params) // 6 > self._compiled_template_kwargs[
            'nr_projs_global']:
            raise ValueError(
                f"Number of projections, {len(params)}, exceeds the "
                f"the maximum of the compiled kernel, namely "
                f"{self._compiled_template_kwargs['nr_projs_global']}. "
                f"Please recompile the kernel with a higher "
                f"`max_projs`.")
        copy_to_symbol(self._module, 'params', params)
        self._params_are_set = True

    def __call__(self,
                 texture: Any,
                 volume: Any,
                 volume_geometry: VolumeGeometry):
        """XrayBackprojection with conebeam geometry.

        Parameters
        ----------
        texture : list[TextureObject] or TextureObject
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
                    f"there is only support for dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`volume` must be a CuPy ndarray.")
        assert volume.flags.c_contiguous is True, (
            f"`{self.__class__.__name__}` is not tested without "
            f"C-contiguous data.")

        if not volume_geometry.has_isotropic_voxels():
            raise NotImplementedError(
                f"`{self.__class__.__name__}` is not tested with anisotropic "
                f"voxels yet.")

        if not self.is_compiled:
            raise KernelNotCompiledException(
                "Please compile the kernel with `compile()`.")

        if isinstance(texture, TextureObject):
            assert texture.ResDesc.cuArr is not None, (
                "A single texture object for all the projections has been "
                "passed, but it does not contain a CUDA array attached.")
            cuArr = texture.ResDesc.cuArr
            dims = cuArr.depth, cuArr.height, cuArr.width
            nr_projs = dims[self._proj_axs[0]]
        else:
            raise ValueError("Please give in a single TextureObject")

        if not self._params_are_set:
            raise KernelMemoryUnpreparedException(
                "Please set the parameters with `set_params()`.")

        fan_bp = self._module.get_function("fan_bp")
        volume_shape = volume_geometry.shape[:2]
        vox_vol = cp.float32(volume_geometry.voxel_volume())
        blocks = np.ceil(np.asarray(volume_shape)
                         / self._vox_block).astype(np.int32)
        for start in range(0, nr_projs, self._projs_block):
            fan_bp((blocks[0], blocks[1]),  # grid
                   (self._vox_block[0], self._vox_block[1]),  # threads
                   (volume,
                    texture,
                    start,
                    nr_projs,
                    volume_shape[1],
                    volume_shape[0],
                    volume_shape[0],  # TODO(Adriaan): support pitched volumes?
                    vox_vol))

    @staticmethod
    def geoms2params(projection_geometry, volume_geometry: VolumeGeometry,
                     with_fbp=False):
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
          || x (s-d) || + ||s d|| / || u (s-x) ||
        - ray density weighting factor for the adjoint
          || u (s-d) || / ( |u| * || u (s-x) || )
        - fan-beam FBP weighting factor
          ( || u s || / || u (s-x) || ) ^ 2
        """
        if isinstance(projection_geometry, list):
            geom_seq = GeometrySequence.fromList(projection_geometry)
        else:
            geom_seq = copy.deepcopy(projection_geometry)

        xp = geom_seq.xp
        normalize_(geom_seq, volume_geometry)
        u = geom_seq.u * geom_seq.detector.pixel_width[..., xp.newaxis]
        s = geom_seq.source_position
        d = geom_seq.detector_extent_min

        def _det2(x, y):
            return x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]

        if not with_fbp:
            # goal: 1/fDen = || u (s-d) || / ( |u| * || u (s-x) || )
            # fDen = ( |u| * || u (s-x) || ) / || u (s-d) ||
            # i.e. scale = |u| /  || u (s-d) ||
            scale = xp.linalg.norm(u) / _det2(u, s - d)
        else:
            # goal: 1/fDen = || u s || / || u (s-x) ||
            # fDen = || u (s-x) || / || u s ||
            # i.e., scale = 1 / || u s ||
            scale = 1. / _det2(u, s)

        numC = scale * _det2(s, d)
        numX = scale * (s - d)[:, 1]
        numY = -scale * (s - d)[:, 0]
        denC = scale * _det2(u, s)  # == 1.0 for FBP
        denX = scale * u[:, 1]
        denY = -scale * u[:, 0]
        return cp.asarray([numC, numX, numY, denC, denX, denY]).flatten()
