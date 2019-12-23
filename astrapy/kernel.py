from abc import ABC
from pathlib import Path
import cupy as cp
import numpy as np
import cupy.cuda.texture as txt
from cupy.cuda.runtime import cudaChannelFormatKindFloat, cudaResourceTypeArray, \
    cudaReadModeElementType, cudaAddressModeBorder, cudaFilterModeLinear

import astrapy

MAX_ANGLES = 2560  # but why


def cupy_get_texture_obj(obj):
    # @todo how do I choose these values?
    channel_desc = txt.ChannelFormatDescriptor(32, 0, 0, 0, cudaChannelFormatKindFloat)

    # @todo it looks as if my definition of width and height are reversed, do I need Fortran order?
    # or is this maybe an inconsistency in the Cupy interface?
    volume_texture_array = txt.CUDAarray(channel_desc, obj.shape[1], obj.shape[0])
    volume_texture_array.copy_from(obj)

    resource_desc = txt.ResourceDescriptor(
        cudaResourceTypeArray, cuArr=volume_texture_array)

    texture_desc = txt.TextureDescriptor(
        (cudaAddressModeBorder, cudaAddressModeBorder),  # address modes
        cudaFilterModeLinear,  # filter modebase = {NoneType} None
        cudaReadModeElementType)

    return txt.TextureObject(resource_desc, texture_desc)


class Kernel(ABC):
    FLOAT_DTYPE = cp.float32
    SUPPORTED_DTYPES = [cp.float32]

    def __init__(self, path: str, function: str):
        fan_fp_module = cp.RawModule(Path(path).read_text())
        self.kernel = fan_fp_module.get_function(function)


class FanProjection(Kernel):
    def __init__(self, path: str = "../kernels/fan_fp.cu", function: str = "fan_fp"):
        super().__init__(path, function)

        self.ANGLES_PER_BLOCK = 16
        self.DET_BLOCK_SIZE = 32  # @todo
        self.BLOCK_SLICES = 64  # why o why

    def __call__(self, volume: astrapy.Volume, sino: astrapy.Sinogram, angles,
                 rays_per_pixel: int = 1
                 ) -> astrapy.Sinogram:
        """Forward projection with fan geometry."""

        if isinstance(sino.data, np.ndarray):
            sino.data = cp.asarray(sino.data, dtype=self.FLOAT_DTYPE)
        elif isinstance(sino.data, cp.ndarray):
            if sino.data.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`projections` must be a 2-dimensional ndarray.")

        if sino.ndim != 2:
            raise ValueError("`projections` must have exactly 2 dimensions.")

        if isinstance(volume.data, np.ndarray):
            volume.data = cp.asarray(volume.data, dtype=self.FLOAT_DTYPE)
        elif isinstance(volume.data, cp.ndarray):
            if volume.data.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`volume` must be a Numpy or CuPy ndarray.")

        if volume.ndim != 2:
            raise ValueError("`volume` must have exactly 2 dimensions.")

        if not volume.has_isotropic_voxels:
            raise NotImplementedError("`FanFP` is not tested for anisotropic voxels yet.")

        # convert angles so that detector position is not the center point but the edge point
        # @todo this should be abstracted away as an operation on a geometry
        angles_converted = [None] * len(angles)
        nr_pixels = sino.data.shape[1]

        for i, angle in enumerate(angles):
            converted_angle = angle
            det_pos = np.array([angle.det_x, angle.det_y])
            det_direction = np.array([angle.det_u_x, angle.det_u_y])
            det_pos = det_pos - det_direction * nr_pixels / 2
            converted_angle.det_x = det_pos[0]
            converted_angle.det_y = det_pos[1]
            angles_converted[i] = converted_angle.to_list()

        # upload angles to GPU
        angles = cp.asarray(angles_converted, dtype=self.FLOAT_DTYPE)

        # initialize @todo what is the performance here, can we do this faster?
        sino.data.fill(0.)

        # chunk arrays in blocks of MAX_ANGLES
        for angle_start in range(0, len(angles), MAX_ANGLES):
            angle_end = min(angle_start + MAX_ANGLES, len(angles))
            self._call_chunk(volume, sino.data, angles[angle_start:angle_end], rays_per_pixel)

        return sino

    def _call_chunk(self, volume: astrapy.Volume, projections, angles, rays_per_pixel):
        nr_pixels = projections.shape[1]
        output_scale = volume.voxel_size[0]
        angles /= volume.voxel_size[0]

        # @todo transfer angles to constant memory
        block_x, block_y = self.DET_BLOCK_SIZE, self.ANGLES_PER_BLOCK
        angle_block_start, angle_block_end = 0, len(angles)
        grid_x = (angle_block_end - angle_block_start + self.ANGLES_PER_BLOCK - 1) // self.ANGLES_PER_BLOCK
        grid_y = (nr_pixels + self.DET_BLOCK_SIZE - 1) // self.DET_BLOCK_SIZE  # angle blocks, regions

        # @todo using pinned memory?
        device = cp.cuda.Device()
        # memory_pool = cp.cuda.PinnedMemoryPool()
        # cp.cuda.set_pinned_memory_allocator(memory_pool.malloc)

        streams = []
        stream1 = cp.cuda.stream.Stream()
        streams.append(stream1)

        volume_texture = cupy_get_texture_obj(volume.data)

        with stream1:
            for start_slice in range(0, volume_texture.ResDesc.cuArr.width, self.BLOCK_SLICES):
                self.kernel(
                    (grid_x, grid_y),
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
                     cp.bool(True),
                     cp.float32(output_scale)
                     ))

        stream2 = cp.cuda.stream.Stream()
        streams.append(stream2)

        with stream2:
            for start_slice in range(0, volume_texture.ResDesc.cuArr.height, self.BLOCK_SLICES):
                self.kernel(
                    (grid_x, grid_y),
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
                     cp.bool(False),
                     cp.float32(output_scale)
                     ))

        del stream1
        del stream2

        #     cudaThreadSynchronize();
        device.synchronize()
        #     cudaTextForceKernelsCompletion();
        #     cudaFreeArray(D_dataArray);
        # @todo free arrays?


class FanBackprojection(Kernel):
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
            return [self.num_c, self.num_x, self.num_y, self.den_c, self.den_x, self.den_y]

    def __init__(self, path: str = "../kernels/fan_bp.cu", function="fan_bp"):
        super().__init__(path, function)

        self.BLOCK_SLICES = 16
        self.BLOCK_SLICE_SIZE = 32
        self.ANGLES_PER_BLOCK = 16

    def __call__(self, volume: astrapy.Volume, sino: astrapy.Sinogram, angles) -> astrapy.Volume:
        """Forward projection with fan geometry."""

        if isinstance(sino.data, np.ndarray):
            sino.data = cp.asarray(sino.data, dtype=self.FLOAT_DTYPE)
        elif isinstance(sino.data, cp.ndarray):
            if sino.data.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`sino` must be a 2-dimensional ndarray.")

        if sino.data.ndim != 2:
            raise ValueError("`sino` must have exactly 2 dimensions.")

        if isinstance(volume.data, np.ndarray):
            volume.data = cp.asarray(volume.data, dtype=self.FLOAT_DTYPE)
        elif isinstance(volume.data, cp.ndarray):
            if volume.data.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`volume` must be a Numpy or CuPy ndarray.")

        if volume.ndim != 2:
            raise ValueError("`volume` must have exactly 2 dimensions.")

        if not volume.has_isotropic_voxels:
            # I think scaling WILL go wrong (i.e. ASTRA fails when pixels are not square)
            raise NotImplementedError("`FanFB` is not tested for anisotropic voxels yet.")

        # initialize, @todo can this be done faster?
        volume.data.fill(0.)

        # convert angles so that detector position is not the center point but the edge point
        # @todo this should be abstracted away as an operation on a geometry
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

        params = np.array([a.to_list() for a in self._angles_to_params(angles_converted)])
        params = cp.array(params, dtype=self.FLOAT_DTYPE)

        # @todo refactor into a "chunking method", to share logic with fan_fp
        # chunk arrays in blocks of MAX_ANGLES
        for angle_start in range(0, len(angles), MAX_ANGLES):
            angle_end = min(angle_start + MAX_ANGLES, len(angles))
            self._call_chunk(volume.data, proj_texture, params[angle_start:angle_end], scaling_factor)

        # angle_extent = self.geometry.motion_partition.extent
        # num_angles = self.geometry.motion_partition.shape
        # scaling_factor = (angle_extent / num_angles).prod()

        # todo: nasty stuff, what if I don't have equidistant angles?
        # I took this from ODL, but I'm not sure if they tested for arbitrary angles
        scaling_factor = 2 * np.pi / 360

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
        assert(len(params) < MAX_ANGLES)

        block_x, block_y = self.BLOCK_SLICES, self.BLOCK_SLICE_SIZE
        grid_x = (volume.shape[0] + self.BLOCK_SLICES - 1) // self.BLOCK_SLICES
        grid_y = (volume.shape[1] + self.BLOCK_SLICE_SIZE - 1) // self.BLOCK_SLICE_SIZE

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
            scale = np.linalg.norm(pixel_vector) / np.linalg.det((pixel_vector, tube - detector))

            p = self._Param(scale * np.linalg.det((tube, detector)),
                            scale * (tube - detector)[1],
                            -scale * (tube - detector)[0],
                            scale * np.linalg.det((pixel_vector, tube)),
                            scale * pixel_vector[1],
                            -scale * pixel_vector[0])
            fan_params.append(p)

        return fan_params
