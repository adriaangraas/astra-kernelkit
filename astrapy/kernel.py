from abc import ABC, abstractmethod
from pathlib import Path

import cupy as cp
import cupy.cuda.texture as txt
import numpy as np
from cupy.cuda.runtime import cudaAddressModeBorder, \
    cudaChannelFormatKindFloat, cudaFilterModeLinear, cudaReadModeElementType, \
    cudaResourceTypeArray
from jinja2 import Template

import astrapy
from astrapy import FlatDetector


def cupy_get_texture_obj(obj):
    """Creates a 2D / 3D texture of float format (todo: precision?)
    """
    assert obj.ndim in [2, 3]

    # TODO: how do I choose these values?
    channel_desc = txt.ChannelFormatDescriptor(32, 0, 0, 0,
                                               cudaChannelFormatKindFloat)

    # TODO: it looks as if my definition of width and height are reversed, do I need Fortran order?
    # or is this maybe an inconsistency in the Cupy interface?
    volume_texture_array = txt.CUDAarray(channel_desc,
                                         *reversed(obj.shape))  # TODO: 3D?
    volume_texture_array.copy_from(obj)

    resource_desc = txt.ResourceDescriptor(cudaResourceTypeArray,
                                           cuArr=volume_texture_array)

    texture_desc = txt.TextureDescriptor(
        [cudaAddressModeBorder] * obj.ndim,
        # address modes for all dims of the object
        cudaFilterModeLinear,  # filter modebase = {NoneType} None
        cudaReadModeElementType)

    return txt.TextureObject(resource_desc, texture_desc)


class Kernel(ABC):
    FLOAT_DTYPE = cp.float32
    SUPPORTED_DTYPES = [cp.float32]

    def __init__(self, path: str, function: str):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


def load_jinja2_kernel(path: str, function: str, **template_kwargs):
    """Renders Jinja2 template and imports kernel in CuPy"""
    template = Template(Path(path).read_text())
    rendered = template.render(**template_kwargs)
    module = RawModule(rendered)
    return module.get_function(function)


class FanProjection(Kernel):
    """Fanbeam forward projection"""
    ANGLES_PER_BLOCK = 16
    DET_BLOCK_SIZE = 32  # @todo
    BLOCK_SLICES = 64  # why o why
    MAX_ANGLES = 2560  # but why

    def __init__(self, path: str = "../kernels/fan_fp.cu",
                 function: str = "fan_fp"):
        super().__init__(path, function)

        self.horizontal = load_jinja2_kernel(
            path, function,
            angles_per_block=self.ANGLES_PER_BLOCK,
            det_block_size=self.DET_BLOCK_SIZE,
            block_slices=self.BLOCK_SLICES,
            mode_horizontal=True)

        self.vertical = load_jinja2_kernel(
            path, function,
            angles_per_block=self.ANGLES_PER_BLOCK,
            det_block_size=self.DET_BLOCK_SIZE,
            block_slices=self.BLOCK_SLICES,
            mode_horizontal=False)

    def __call__(self,
                 volume: astrapy.Volume,
                 sino: astrapy.Sinogram,
                 geometry: astrapy.GeometryDict,
                 # TODO(Adriaan): check if subtype is 2D geometry
                 rays_per_pixel: int = 1) -> astrapy.Sinogram:
        """Forward projection with fan geometry."""

        if isinstance(sino.data, np.ndarray):
            sino.data = cp.asarray(sino.data, dtype=self.FLOAT_DTYPE)
        elif isinstance(sino.data, cp.ndarray):
            if sino.data.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`projections` must be a ndarray.")

        if sino.ndim != 2:
            raise ValueError("`projections` must have exactly 2 dimensions.")

        if isinstance(volume.data, np.ndarray):
            volume.data = cp.asarray(volume.data, dtype=self.FLOAT_DTYPE)
        elif isinstance(volume.data, cp.ndarray):
            if volume.data.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`volume` must be a Numpy or CuPy ndarray.")

        if volume.ndim != 2:
            raise ValueError("`volume` must have exactly 2 dimensions.")

        if not volume.has_isotropic_voxels:
            raise NotImplementedError(
                "`FanProjection` is not tested with anisotropic voxels yet.")

        # convert angles so that detector position is not the center point but the edge point
        # todo: this should be abstracted away as an operation on a geometry
        nr_pixels = geometry.detector(
            0).nr_pixels  # todo: now t=0, we need a guarantee that the detector does not change
        # nr_pixels = sino.data.shape[1]

        vectors = []
        for i, geom in enumerate(geometry):  # type: (int, astrapy.Static2DGeometry)
            # find the starting point on the side of the detector
            horizontal_pixel_vector = np.multiply(
                geom.detector.pixel_width,
                FlatDetector.AXIS_HORIZONTAL)
            det_direction = geom.detector_rotation_matrix @ horizontal_pixel_vector
            left_detector_position = (
                    geom.detector_position -
                    det_direction * nr_pixels / 2)

            # convert to vector description
            vectors.append(
                [*left_detector_position,
                 *det_direction,
                 *geom.tube_position])

        # upload angles to GPU
        geometry = cp.asarray(vectors, dtype=self.FLOAT_DTYPE)

        # initialize @todo what is the performance here, can we do this faster?
        sino.data.fill(0.)

        volume_texture = cupy_get_texture_obj(volume.data)

        output_scale = volume.voxel_size[0]

        # chunk arrays in blocks of MAX_ANGLES
        for angle_start in range(0, len(geometry), self.MAX_ANGLES):
            angle_end = min(angle_start + self.MAX_ANGLES, len(geometry))
            self._call_chunk(volume_texture,
                             sino.data,
                             geometry[angle_start:angle_end],
                             rays_per_pixel,
                             output_scale)

        return sino

    def _call_chunk(self,
                    volume_texture: txt.TextureObject,
                    projections,
                    angles,
                    rays_per_pixel: int,
                    output_scale: float):
        nr_pixels = projections.shape[1]
        angles /= output_scale

        # @todo transfer angles to constant memory
        block_x, block_y = self.DET_BLOCK_SIZE, self.ANGLES_PER_BLOCK
        angle_block_start, angle_block_end = 0, len(angles)
        grid_x = (
                     angle_block_end - angle_block_start + self.ANGLES_PER_BLOCK - 1) // self.ANGLES_PER_BLOCK
        grid_y = (
                     nr_pixels + self.DET_BLOCK_SIZE - 1) // self.DET_BLOCK_SIZE  # angle blocks, regions

        # @todo using pinned memory?
        device = cp.cuda.Device()
        # memory_pool = cp.cuda.PinnedMemoryPool()
        # cp.cuda.set_pinned_memory_allocator(memory_pool.malloc)

        # we have two streams, one for each kernel
        streams = [cp.cuda.stream.Stream()] * 2
        kernels = (self.horizontal, self.vertical)

        for stream, kernel in zip(streams, kernels):
            # note that the kernels are executed asynchronously
            with stream:
                for start_slice in range(0, volume_texture.ResDesc.cuArr.width,
                                         self.BLOCK_SLICES):
                    kernel((grid_x, grid_y),
                           (block_x, block_y),
                           (volume_texture,
                            projections,
                            angles,
                            start_slice,
                            angle_block_start,
                            angle_block_end,
                            nr_pixels,
                            rays_per_pixel,
                            volume_texture.ResDesc.cuArr.width,
                            volume_texture.ResDesc.cuArr.height,
                            cp.float32(output_scale)))

        #     cudaThreadSynchronize();
        device.synchronize()
        #     cudaTextForceKernelsCompletion();
        #     cudaFreeArray(D_dataArray);
        # @todo free arrays?


class FanBackprojection(Kernel):
    BLOCK_SLICES = 16
    BLOCK_SLICE_SIZE = 32
    ANGLES_PER_BLOCK = 16
    MAX_ANGLES = 2560  # but why

    class _Param:
        """Inner class to support kernel parameter computations."""

        def __init__(self, num_c, num_x, num_y, den_c, den_x, den_y):
            self.num_c = num_c
            self.num_x = num_x
            self.num_y = num_y
            self.den_c = den_c
            self.den_x = den_x
            self.den_y = den_y

        def to_list(self):
            return [self.num_c, self.num_x, self.num_y, self.den_c, self.den_x,
                    self.den_y]

    def __init__(self, path: str = "../kernels/fan_bp.cu", function="fan_bp"):
        self.kernel = load_jinja2_kernel(path, function,
                                         block_slices=self.BLOCK_SLICES,
                                         block_slice_size=self.BLOCK_SLICE_SIZE,
                                         angles_per_block=self.ANGLES_PER_BLOCK).kernel

    def __call__(self,
                 volume: astrapy.Volume,
                 sino: astrapy.Sinogram,
                 angles) -> astrapy.Volume:
        """Forward projection with fan geometry."""
        if isinstance(sino.data, np.ndarray):
            sino.data = cp.asarray(sino.data, dtype=self.FLOAT_DTYPE)
        elif isinstance(sino.data, cp.ndarray):
            if sino.data.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`sino` must be a 2-dimensional ndarray.")

        if sino.data.ndim != 2:
            raise ValueError("`sino` must have exactly 2 dimensions.")

        if isinstance(volume.data, np.ndarray):
            volume.data = cp.asarray(volume.data, dtype=self.FLOAT_DTYPE)
        elif isinstance(volume.data, cp.ndarray):
            if volume.data.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`volume` must be a Numpy or CuPy ndarray.")

        if volume.ndim != 2:
            raise ValueError("`volume` must have exactly 2 dimensions.")

        if not volume.has_isotropic_voxels:
            # I think scaling WILL go wrong (i.e. ASTRA Toolbox cannot do non-square pixels as well)
            raise NotImplementedError(
                "`FanBackprojection` is not tested with anisotropic voxels yet.")

        # initialize, @todo can this be done faster?
        volume.data.fill(0.)

        # convert angles so that detector position is not the center point but the edge point
        # TODO: this should be abstracted away as an operation on a geometry
        angles_converted = [None] * len(angles)
        nr_pixels = sino.data.shape[1]

        scaling_factor = volume.voxel_size[0]
        for i, angle in enumerate(angles):
            converted_angle = angle
            det_pos = np.array([angle.det_x, angle.det_y])
            det_direction = np.array([angle.det_u_x, angle.det_u_y])
            det_pos = det_pos - det_direction * nr_pixels / 2
            converted_angle.det_x = det_pos[0]
            converted_angle.det_y = det_pos[1]

            # then shift to the center of the geom?
            dx = -(volume.extent_min[0] + volume.extent_max[0]) / 2
            dy = -(volume.extent_min[1] + volume.extent_max[1]) / 2
            converted_angle.tube_x += dx
            converted_angle.tube_y += dy
            converted_angle.det_x += dx
            converted_angle.det_y += dy

            # scale appropriately (lol why I don't get it?)
            pre_scaling = 1 / scaling_factor
            converted_angle.tube_x *= pre_scaling
            converted_angle.tube_y *= pre_scaling
            converted_angle.det_x *= pre_scaling
            converted_angle.det_y *= pre_scaling
            converted_angle.det_u_x *= pre_scaling
            converted_angle.det_u_y *= pre_scaling

            angles_converted[i] = converted_angle

        proj_texture = cupy_get_texture_obj(sino.data)

        params = np.array(
            [a.to_list() for a in self._angles_to_params(angles_converted)])
        params = cp.array(params, dtype=self.FLOAT_DTYPE)

        # TODO: refactor into a "chunking method", to share logic with fan_fp
        # chunk arrays in blocks of MAX_ANGLES
        for angle_start in range(0, len(angles), self.MAX_ANGLES):
            angle_end = min(angle_start + self.MAX_ANGLES, len(angles))
            self._call_chunk(volume.data, proj_texture,
                             params[angle_start:angle_end], scaling_factor)

        # angle_extent = self.geometry.motion_partition.extent
        # num_angles = self.geometry.motion_partition.shape
        # scaling_factor = (angle_extent / num_angles).prod()

        # todo: nasty stuff, what if I don't have equidistant angles?
        # I took this from ODL, but I'm not sure if they tested for arbitrary angles
        scaling_factor = 2 * np.pi / 360  # todo

        # Correct in case of non-weighted spaces
        # proj_extent = float(self.proj_space.partition.extent.prod())
        # proj_size = float(self.proj_space.partition.size)

        # proj_weighting = proj_extent / proj_size
        # scaling_factor *= (self.proj_space.weighting.const /
        #                    proj_weighting)
        # scaling_factor /= (self.reco_space.weighting.const /
        #                    self.reco_space.cell_volume)

        scaling_factor /= float(volume.voxel_volume)
        scaling_factor *= float(sino.pixel_volume)

        volume.data *= scaling_factor

        return volume

    def _call_chunk(self, volume, proj_texture, params, output_scale: float):
        assert (len(params) < self.MAX_ANGLES)

        block_x, block_y = self.BLOCK_SLICES, self.BLOCK_SLICE_SIZE
        grid_x = (volume.shape[0] + self.BLOCK_SLICES - 1) // self.BLOCK_SLICES
        grid_y = (volume.shape[
                      1] + self.BLOCK_SLICE_SIZE - 1) // self.BLOCK_SLICE_SIZE

        with cp.cuda.stream.Stream():
            for start_angle in range(0, len(params), self.ANGLES_PER_BLOCK):
                self.kernel(
                    (grid_x, grid_y),
                    (block_x, block_y),
                    (proj_texture,
                     volume,
                     params,
                     start_angle,
                     len(params),
                     volume.shape[0],
                     volume.shape[1],
                     cp.float32(output_scale)
                     ))

        cp.cuda.Device().synchronize()

    def _angles_to_params(self, angles):
        """We need three values in the kernel:
         - projected coordinates of pixels on the detector:
           || x (s-d) || + ||s d|| / || u (s-x) ||
         - ray density weighting factor for the adjoint
           || u (s-d) || / ( |u| * || u (s-x) || )
         - fan-beam FBP weighting factor
           ( || u s || / || u (s-x) || ) ^ 2
        """

        fan_params = []
        for i, angle in enumerate(angles):
            tube = np.array((angle.tube_x, angle.tube_y))
            detector = np.array((angle.det_x, angle.det_y))
            pixel_vector = np.array((angle.det_u_x, angle.det_u_y))

            # goal: 1/fDen = || u s || / || u (s-x) ||
            # fDen = || u (s-x) || / || u s ||
            # i.e., scale = 1 / || u s ||
            # scale = 1.0 / det(u, s)
            scale = np.linalg.norm(pixel_vector) / np.linalg.det(
                (pixel_vector, tube - detector))

            p = self._Param(scale * np.linalg.det((tube, detector)),
                            scale * (tube - detector)[1],
                            -scale * (tube - detector)[0],
                            scale * np.linalg.det((pixel_vector, tube)),
                            scale * pixel_vector[1],
                            -scale * pixel_vector[0])
            fan_params.append(p)

        return fan_params

# class ConeProjection(Kernel):
#     MAX_ANGLES = 1024
#
#     def __init__(self, path: str = "../kernels/cone_fp.cu", function: str = "cone_fp"):
#         self.kernel_X = load_jinja2_kernel(path, function,
#                                            type_coord='DIR_X')
#         self.kernel_Y = load_jinja2_kernel(path, function,
#                                            type_coord='DIR_Y')
#         self.kernel_Z = load_jinja2_kernel(path, function,
#                                            type_coord='DIR_Z')
#
#     def __call__(self, volume: astrapy.Volume, sino: astrapy.Sinogram, angles,
#                  rays_per_pixel: int = 1
#                  ) -> astrapy.Sinogram:
#         """Forward projection with fan geometry."""
#
#         if isinstance(sino.data, np.ndarray):
#             sino.data = cp.asarray(sino.data, dtype=self.FLOAT_DTYPE)
#         elif isinstance(sino.data, cp.ndarray):
#             if sino.data.dtype not in self.SUPPORTED_DTYPES:
#                 raise NotImplementedError(f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
#         else:
#             raise TypeError("`projections` must be a ndarray.")
#
#         if sino.ndim != 3:
#             raise ValueError("`projections` must have exactly 3 dimensions.")
#
#         if isinstance(volume.data, np.ndarray):
#             volume.data = cp.asarray(volume.data, dtype=self.FLOAT_DTYPE)
#         elif isinstance(volume.data, cp.ndarray):
#             if volume.data.dtype not in self.SUPPORTED_DTYPES:
#                 raise NotImplementedError(f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
#         else:
#             raise TypeError("`volume` must be a Numpy or CuPy ndarray.")
#
#         if volume.ndim != 3:
#             raise ValueError("`volume` must have exactly 3 dimensions.")
#
#         if not volume.has_isotropic_voxels:
#             raise NotImplementedError("`ConeProjection` is not tested with anisotropic voxels yet.")
#
#         # convert angles so that detector position is not the center point but the edge point
#         # TODO this should be abstracted away as an operation on a geometry
#         angles_converted = [None] * len(angles)
#         nr_pixels = sino.data.shape[1]
#
#         for i, angle in enumerate(angles):
#             converted_angle = angle
#             det_pos = np.array([angle.det_x, angle.det_y])
#             det_direction = np.array([angle.det_u_x, angle.det_u_y])
#             det_pos = det_pos - det_direction * nr_pixels / 2
#             converted_angle.det_x = det_pos[0]
#             converted_angle.det_y = det_pos[1]
#             angles_converted[i] = converted_angle.to_list()
#
#         # upload angles to GPU
#         angles = cp.asarray(angles_converted, dtype=self.FLOAT_DTYPE)
#
#         # initialize @todo what is the performance here, can we do this faster?
#         sino.data.fill(0.)
#
#         volume_texture = cupy_get_texture_obj(volume.data)
#
#         output_scale = volume.voxel_size[0]  # todo: this is from 2D case
#
#         # chunk arrays in blocks of MAX_ANGLES
#         for angle_start in range(0, len(angles), self.MAX_ANGLES):
#             angle_end = min(angle_start + self.MAX_ANGLES, len(angles))
#             self._call_chunk(volume_texture, sino.data, angles[angle_start:angle_end], rays_per_pixel, output_scale)
#
#         # todo: want to free texture memory? I don't think so
#         #         cudaFreeArray(cuArray);
#         return sino
#
#     #             cudaPitchedPtr D_subprojData = D_projData;
#     #             D_subprojData.ptr = (char *) D_projData.ptr + iAngle * D_projData.pitch;
#
#     #             ret = ConeFP_Array_internal(D_subprojData,
#     #                                         dims, iEndAngle - iAngle, angles + iAngle,
#     #                                         params);
#
#     def _call_chunk(self, volume_texture: txt.TextureObject, sino, angles: list, rays_per_pixel: int,
#                     output_scale: float):
#         #     bool ConeFP_Array_internal(cudaPitchedPtr D_projData,
#         #                                const SDimensions3D &dims, unsigned int angleCount, const SConeProjection *angles,
#         #                                const SProjectorParams3D &params) {
#         #         // transfer angles to constant memory
#
#         #         dim3 dimBlock(g_detBlockU, g_anglesPerBlock); // region size, angles
#
#         # Run over all angles, grouping them into groups of the same
#         # orientation (roughly horizontal vs. roughly vertical).
#         # Start a stream of grids for each such group.
#
#         block_start = 0
#         block_end = 0
#         block_direction = 0
#
#         cube = True
#         if (abs(fVolScaleX / fVolScaleY - 1.0) > 0.00001):
#             cube = False
#
#         if (abs(fVolScaleX / fVolScaleZ - 1.0) > 0.00001):
#             cube = False
#
#         SCALE_CUBE
#         scube
#         scube.fOutputScale = params.fOutputScale * params.fVolScaleX
#
#         SCALE_NONCUBE
#         snoncubeX
#         fS1 = params.fVolScaleY / params.fVolScaleX
#         snoncubeX.fScale1 = fS1 * fS1
#         fS2 = params.fVolScaleZ / params.fVolScaleX
#         snoncubeX.fScale2 = fS2 * fS2
#         snoncubeX.fOutputScale = params.fOutputScale * params.fVolScaleX
#
#         SCALE_NONCUBE
#         snoncubeY
#         fS1 = params.fVolScaleX / params.fVolScaleY
#         snoncubeY.fScale1 = fS1 * fS1
#         fS2 = params.fVolScaleZ / params.fVolScaleY
#         snoncubeY.fScale2 = fS2 * fS2
#         snoncubeY.fOutputScale = params.fOutputScale * params.fVolScaleY
#
#         SCALE_NONCUBE
#         snoncubeZ
#         fS1 = params.fVolScaleX / params.fVolScaleZ
#         snoncubeZ.fScale1 = fS1 * fS1
#         fS2 = params.fVolScaleY / params.fVolScaleZ
#         snoncubeZ.fScale2 = fS2 * fS2
#         snoncubeZ.fOutputScale = params.fOutputScale * params.fVolScaleZ
#
#         for a in range(angleCount + 1):
#             dir = -1
#
#             if a != angleCount:
#                 dX = abs(angles[a].fSrcX - (angles[a].fDetSX + dims.iProjU * angles[a].fDetUX * 0.5f +
#                 dims.iProjV * angles[a].fDetVX * 0.5f))
#
#                 dY = fabsf(angles[a].fSrcY - (angles[a].fDetSY + dims.iProjU * angles[a].fDetUY * 0.5f +
#                 dims.iProjV * angles[a].fDetVY * 0.5f))
#
#                 dZ = fabsf(angles[a].fSrcZ - (angles[a].fDetSZ + dims.iProjU * angles[a].fDetUZ * 0.5f +
#                 dims.iProjV * angles[a].fDetVZ * 0.5f))
#
#                 if dX >= dY and dX >= dZ:
#                     dir = 0
#                 elif dY >= dX and dY >= dZ:
#                     dir = 1
#                 else:
#                     dir = 2
#
#             if a == angleCount or dir != blockDirection:
#                 # block done
#                 blockEnd = a
#
#                 if blockStart != blockEnd:
#                     dim3
#                     dimGrid(
#                         ((dims.iProjU + g_detBlockU - 1) / g_detBlockU) *
#                         ((dims.iProjV + g_detBlockV - 1) / g_detBlockV),
#                         (blockEnd - blockStart + g_anglesPerBlock - 1) / g_anglesPerBlock)
#                     # TODO: check if we can't immediately
#                     # destroy the stream after use
#
#                     with cp.cuda.stream.Stream():
#                         if block_direction == 0:
#                             for i in range(0, dims.iVolX, self.BLOCK_SLICES):
#                                 if rays_per_pixel == 1:
#                                     self.kernel_X(
#                                         (dimGrid),
#                                         dimBlock,
#                                         stream)
#                                     D_projData.ptr,
#                                     D_projData.pitch / sizeof(float),
#                                     i,
#                                     blockStart,
#                                     blockEnd,
#                                     dims, scube if cube else snonCube)
#                                     else:
#                                     # cone_FP_SS_t<DIR_X> << < dimGrid, dimBlock, 0, stream >> > ((float *) D_projData.ptr,
#                                     #         D_projData.pitch /
#                                     #         sizeof(float), i, blockStart, blockEnd, dims, params.iRaysPerDetDim, snoncubeX)
#                                     raise NotImplemented("Detector supersampling is currently not supported.")
#
#                         elif block_direction == 1:
#                             for i in range(0, dims.iVolY, self.BLOCK_SLICES):
#                         #                             if (params.iRaysPerDetDim == 1)
#                         #                                 if (cube)
#                         #                                     cone_FP_t<DIR_Y> << < dimGrid, dimBlock, 0, stream >> > ((float *) D_projData.ptr,
#                         #                                             D_projData.pitch / sizeof(float), i, blockStart, blockEnd, dims, scube);
#                         #                                 else
#                         #                                     cone_FP_t<DIR_Y> << < dimGrid, dimBlock, 0, stream >> > ((float *) D_projData.ptr,
#                         #                                             D_projData.pitch / sizeof(float), i, blockStart, blockEnd, dims, snoncubeY);
#                         #                             else
#                         #                                 cone_FP_SS_t<DIR_Y> << < dimGrid, dimBlock, 0, stream >> > ((float *) D_projData.ptr,
#                         #                                         D_projData.pitch /
#                         #                                         sizeof(float), i, blockStart, blockEnd, dims, params.iRaysPerDetDim, snoncubeY);
#                         elif block_direction == 2:
#                             for i in range(0, dims.iVolZ, self.BLOCK_SLICES):
#                     #                             if (params.iRaysPerDetDim == 1)
#                     #                                 if (cube)
#                     #                                     cone_FP_t<DIR_Z> << < dimGrid, dimBlock, 0, stream >> > ((float *) D_projData.ptr,
#                     #                                             D_projData.pitch / sizeof(float), i, blockStart, blockEnd, dims, scube);
#                     #                                 else
#                     #                                     cone_FP_t<DIR_Z> << < dimGrid, dimBlock, 0, stream >> > ((float *) D_projData.ptr,
#                     #                                             D_projData.pitch / sizeof(float), i, blockStart, blockEnd, dims, snoncubeZ);
#                     #                             else
#                     #                                 cone_FP_SS_t<DIR_Z> << < dimGrid, dimBlock, 0, stream >> > ((float *) D_projData.ptr,
#                     #                                         D_projData.pitch /
#                     #                                         sizeof(float), i, blockStart, blockEnd, dims, params.iRaysPerDetDim, snoncubeZ);
#
#                 block_direction = dir
#                 block_start = a
