import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import cupy as cp
import cupy.cuda.texture as txt
import jinja2
import numpy as np
from cupy.core.raw import RawModule
from cupy.cuda.runtime import cudaAddressModeBorder, \
    cudaChannelFormatKindFloat, cudaFilterModeLinear, cudaReadModeElementType, \
    cudaResourceTypeArray

import astrapy
from astrapy import AstraStatic3DGeometry


def _cupy_copy_to_texture(obj) -> txt.TextureObject:
    """Creates a 2D / 3D texture of float format"""
    assert obj.ndim in [2, 3]

    # TODO(Adriaan): how do I choose these values?
    # TODO(Adriaan): do I have a precision choice?
    channel_desc = txt.ChannelFormatDescriptor(32, 0, 0, 0,
                                               cudaChannelFormatKindFloat)

    # TODO: it looks as if my definition of width and height are reversed,
    #  do I need Fortran order? Or is this maybe an inconsistency in
    #  the CUDA/Cupy interface?
    texture_array = txt.CUDAarray(channel_desc,
                                  *reversed(obj.shape))  # TODO: test in 3D
    texture_array.copy_from(obj)

    resource_desc = txt.ResourceDescriptor(cudaResourceTypeArray,
                                           cuArr=texture_array)
    texture_desc = txt.TextureDescriptor(
        [cudaAddressModeBorder] * obj.ndim,
        # address modes for all dims of the object
        cudaFilterModeLinear,  # filter modebase = {NoneType} None
        cudaReadModeElementType)

    return txt.TextureObject(resource_desc, texture_desc)


def _cupy_copy_to_constant(module, name: str, array, dtype=np.float32):
    """Copy array to constant memory
    Inspired from: https://github.com/cupy/cupy/issues/1703
    See also: https://docs.chainer.org/en/v1.3.1/_modules/cupy/creation/from_data.html

    TODO: would it be faster to use `p.copy_from_device` instead?
        We could async pre-upload all angles to a device and copy to
        constant memory from there.
    """
    import ctypes
    p = module.get_global(name)
    a_cpu = np.ascontiguousarray(np.squeeze(array), dtype=dtype)
    p.copy_from(a_cpu.ctypes.data_as(ctypes.c_void_p),
                a_cpu.nbytes)


class _cuda_float4:
    """Helper to encode CUDA float4 in the right order"""

    def __init__(self, x, y, z, w):
        self.x, self.y, self.z, self.w = x, y, z, w

    def to_list(self):
        return [self.x, self.y, self.z, self.w]

    def __str__(self):
        return str(self.to_list())


def load_jinja2_module(path: str,
                       name_expressions=None,
                       **template_kwargs):
    """Renders Jinja2 template and imports kernel in CuPy"""
    template = jinja2.Template(Path(path).read_text(),
                               undefined=jinja2.StrictUndefined)
    rendered = template.render(**template_kwargs)
    module = RawModule(code=rendered,
                       # --std is required for name expressions
                       # TODO: error on c++17, but NVRTC manual supports it?
                       options=('--std=c++11',),
                       name_expressions=name_expressions)
    return module


class Kernel(ABC):
    FLOAT_DTYPE = cp.float32
    SUPPORTED_DTYPES = [cp.float32]

    @abstractmethod
    def __init__(self, path: str):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class FanProjection(Kernel):
    """Fanbeam forward projection"""
    ANGLES_PER_BLOCK = 16
    DET_BLOCK_SIZE = 32
    BLOCK_SLICES = 64
    MAX_ANGLES = 25
    FAN_FP_FUNCTION = "fan_fp"

    def __init__(self, path: str = "../kernels/2d/fan_fp.cu"):
        super().__init__(path)

        # TODO(Adriaan): merge the two modules, like cone_fp
        self.horizontal = load_jinja2_module(
            path,
            angles_per_block=self.ANGLES_PER_BLOCK,
            det_block_size=self.DET_BLOCK_SIZE,
            block_slices=self.BLOCK_SLICES,
            max_angles=self.MAX_ANGLES,
            mode_horizontal=True)

        self.vertical = load_jinja2_module(
            path,
            angles_per_block=self.ANGLES_PER_BLOCK,
            det_block_size=self.DET_BLOCK_SIZE,
            block_slices=self.BLOCK_SLICES,
            max_angles=self.MAX_ANGLES,
            mode_horizontal=False)

    def __call__(self,
                 volume: astrapy.Volume,
                 sino: astrapy.Sinogram,
                 geometry: list[astrapy.Static2DGeometry],
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
                f"`{self.__class__.__name__}` is not tested with anisotropic voxels yet.")

        # convert angles so that detector position is not the center point but the edge point
        # TODO: this should be abstracted away as an operation on a geometry
        nr_pixels = geometry[0].detector.nr_pixels
        # TODO: now t=0, we need a guarantee that the detector does not change
        # nr_pixels = sino.data.shape[1]

        vectors = []
        for geom in geometry:
            # find the starting point on the side of the detector
            # pixel_vector = np.multiply(geom.detector.pixel_width,
            #     astrapy.Flat1DDetector.AXIS_HORIZONTAL)
            # det_direction = geom.detector_rotation_matrix @ pixel_vector
            det_direction = geom.u * geom.detector.pixel_width

            left_detector_position = (
                geom.detector_position -
                det_direction * nr_pixels / 2)

            # convert to vector description
            vectors.append(
                [*left_detector_position,
                 *det_direction,
                 *geom.tube_position])

        # upload angles to GPU
        # plot_geoms = {i: g for i, g in geometry.items() if i < 100 and i % 10 == 0}
        # astrapy.geom2d.plot(plot_geoms)

        # initialize TODO: what is the performance here, can we do this faster?
        sino.data.fill(0.)

        volume_texture = _cupy_copy_to_texture(volume.data)

        output_scale = volume.voxel_size[0]

        # chunk arrays in blocks of MAX_ANGLES
        for angle_start in range(0, len(vectors), self.MAX_ANGLES):
            angle_end = min(angle_start + self.MAX_ANGLES, len(vectors))
            self._call_chunk(volume_texture,
                             sino.data[angle_start:angle_end],
                             vectors[angle_start:angle_end],
                             rays_per_pixel,
                             output_scale)

        return sino

    def _call_chunk(self,
                    volume_texture: txt.TextureObject,
                    projections,
                    angles,
                    rays_per_pixel: int,
                    output_scale: float):
        assert len(projections) == len(angles)

        nr_pixels = projections.shape[1]
        angles /= output_scale

        angle_block_start, angle_block_end = 0, len(angles)
        # angle blocks, regions
        grid_x = (angle_block_end - angle_block_start
                  + self.ANGLES_PER_BLOCK - 1) // self.ANGLES_PER_BLOCK
        grid_y = (nr_pixels
                  + self.DET_BLOCK_SIZE - 1) // self.DET_BLOCK_SIZE

        # TODO use pinned host memory?
        # memory_pool = cp.cuda.PinnedMemoryPool()
        # cp.cuda.set_pinned_memory_allocator(memory_pool.malloc)

        # we have two streams, one for each kernel
        streams = [cp.cuda.stream.Stream()] * 2
        kernels = (self.horizontal.get_function(self.FAN_FP_FUNCTION),
                   self.vertical.get_function(self.FAN_FP_FUNCTION))
        modules = (self.horizontal, self.vertical)

        for stream, kernel, module in zip(streams, kernels, modules):
            _cupy_copy_to_constant(module, 'csrcX', angles[:, 4::6])
            _cupy_copy_to_constant(module, 'csrcY', angles[:, 5::6])
            _cupy_copy_to_constant(module, 'cdetSX', angles[:, 0::6])
            _cupy_copy_to_constant(module, 'cdetSY', angles[:, 1::6])
            _cupy_copy_to_constant(module, 'cdetUX', angles[:, 2::6])
            _cupy_copy_to_constant(module, 'cdetUY', angles[:, 3::6])

            # note that the kernels are executed asynchronously
            with stream:
                for start_slice in range(0, volume_texture.ResDesc.cuArr.width,
                                         self.BLOCK_SLICES):
                    kernel((grid_x, grid_y),
                           (self.DET_BLOCK_SIZE, self.ANGLES_PER_BLOCK),
                           (volume_texture,
                            projections,
                            # angles,
                            start_slice,
                            angle_block_start,
                            angle_block_end,
                            nr_pixels,
                            rays_per_pixel,
                            volume_texture.ResDesc.cuArr.width,
                            volume_texture.ResDesc.cuArr.height,
                            cp.float32(output_scale)))

        cp.cuda.Device().synchronize()


class FanBackprojection(Kernel):
    BLOCK_SLICES = 16
    BLOCK_SLICE_SIZE = 32
    ANGLES_PER_BLOCK = 16
    MAX_ANGLES = 2560
    FAN_BP_FUNCTION = "fan_bp"

    def __init__(self, path: str = "../kernels/2d/fan_bp.cu",
                 function="fan_bp"):
        super().__init__(path)

        self.module = load_jinja2_module(path,
                                         block_slices=self.BLOCK_SLICES,
                                         block_slice_size=self.BLOCK_SLICE_SIZE,
                                         angles_per_block=self.ANGLES_PER_BLOCK,
                                         max_angles=self.MAX_ANGLES)

    def __call__(self,
                 volume: astrapy.Volume,
                 sino: astrapy.Sinogram,
                 geometry: list[astrapy.Static2DGeometry]
                 ) -> astrapy.Volume:
        """Backprojection with fan geometry."""

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
            # I think scaling will go wrong (i.e. ASTRA Toolbox cannot do non-square pixels as well)
            raise NotImplementedError(
                f"`{self.__class__.__name__}` is not tested with anisotropic voxels yet.")

        # initialize, TODO: can this be done faster/is this necessary here?
        volume.data.fill(0.)

        # TODO: circular geometry is assumed, we need to either assert
        #    that the geometry is circular here, or generalize to
        #    non-circular geometries
        # TODO: this should be abstracted away as an operation on a geometry
        converted_geometry = copy.deepcopy(geometry)
        output_scale = volume.voxel_size[0]

        # convert angles so that detector position is not the center point but the edge point
        # TODO: I should use decorator patterns here on geometries
        for ngeom, geom in zip(converted_geometry, geometry):
            ngeom.detector_position[:] = (
                geom.detector_position
                - geom.u * geom.detector.width / 2)

            # then shift to the center of the geom?
            dx = -(volume.extent_min[0] + volume.extent_max[0]) / 2
            dy = -(volume.extent_min[1] + volume.extent_max[1]) / 2
            ngeom.tube_position[:] = ngeom.tube_position + [dx, dy]
            ngeom.detector_position[:] = ngeom.detector_position + [dx, dy]

            # scale appropriately
            s = 1. / output_scale
            ngeom.tube_position[:] = s * ngeom.tube_position[:]
            ngeom.detector_position[:] = s * ngeom.detector_position[:]
            ngeom.detector.pixel_width = s * ngeom.detector.pixel_width

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(sino.data.get())
        # plt.show()
        # astrapy.geom2d.plot(converted_geometry)
        proj_texture = _cupy_copy_to_texture(sino.data)

        params = np.array(
            [a.to_list() for a in self._geoms2params(converted_geometry)])

        # TODO: refactor into a "chunking method", to share logic with fan_fp
        # chunk arrays in blocks of MAX_ANGLES
        for angle_start in range(0, len(geometry), self.MAX_ANGLES):
            angle_end = min(angle_start + self.MAX_ANGLES, len(geometry))
            self._call_chunk(volume.data,
                             proj_texture,
                             params[angle_start:angle_end],
                             output_scale)

        # angle_extent = self.geometry.motion_partition.extent
        # num_angles = self.geometry.motion_partition.shape
        # scaling_factor = (angle_extent / num_angles).prod()

        # TODO: nasty stuff, what if I don't have equidistant angles?
        #   I took this from ODL, but I'm not sure if they tested for arbitrary angles
        output_scale = 2 * np.pi / 360  # TODO

        # Correct in case of non-weighted spaces
        # proj_extent = float(self.proj_space.partition.extent.prod())
        # proj_size = float(self.proj_space.partition.size)

        # proj_weighting = proj_extent / proj_size
        # scaling_factor *= (self.proj_space.weighting.const /
        #                    proj_weighting)
        # scaling_factor /= (self.reco_space.weighting.const /
        #                    self.reco_space.cell_volume)

        output_scale /= float(volume.voxel_volume)
        output_scale *= float(sino.pixel_volume)

        volume.data *= output_scale

        return volume

    def _call_chunk(self, volume, proj_texture, params, output_scale: float):
        assert (len(params) < self.MAX_ANGLES)
        grid_x = (volume.shape[0]
                  + self.BLOCK_SLICES - 1) // self.BLOCK_SLICES
        grid_y = (volume.shape[1]
                  + self.BLOCK_SLICE_SIZE - 1) // self.BLOCK_SLICE_SIZE

        with cp.cuda.stream.Stream():
            _cupy_copy_to_constant(self.module, 'params', params.flatten())
            for start_angle in range(0, len(params), self.ANGLES_PER_BLOCK):
                self.module.get_function(self.FAN_BP_FUNCTION)(
                    (grid_x, grid_y),
                    (self.BLOCK_SLICES, self.BLOCK_SLICE_SIZE),
                    (proj_texture,
                     volume,
                     start_angle,
                     len(params),
                     volume.shape[0],
                     volume.shape[1],
                     cp.float32(output_scale)))

        cp.cuda.Device().synchronize()

    def _geoms2params(self, geoms):
        """We need three values in the kernel:
         - projected coordinates of pixels on the detector:
           || x (s-d) || + ||s d|| / || u (s-x) ||
         - ray density weighting factor for the adjoint
           || u (s-d) || / ( |u| * || u (s-x) || )
         - fan-beam FBP weighting factor
           ( || u s || / || u (s-x) || ) ^ 2
        """

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
                return [self.num_c, self.num_x, self.num_y,
                        self.den_c, self.den_x, self.den_y]

        fan_params = []
        for geom in geoms:
            tube = geom.tube_position
            detector = geom.detector_position
            pixel_vector = geom.u * geom.detector.pixel_width

            # goal: 1/fDen = || u s || / || u (s-x) ||
            # fDen = || u (s-x) || / || u s ||
            # i.e., scale = 1 / || u s ||
            # scale = 1.0 / det(u, s)
            scale = (
                np.linalg.norm(pixel_vector) /
                np.linalg.det((pixel_vector, tube - detector))
            )

            p = _Param(
                num_c=scale * np.linalg.det((tube, detector)),
                num_x=scale * (tube - detector)[1],
                num_y=-scale * (tube - detector)[0],
                den_c=scale * np.linalg.det((pixel_vector, tube)),
                den_x=scale * pixel_vector[1],
                den_y=-scale * pixel_vector[0]
            )
            fan_params.append(p)

        return fan_params


def _prep_geom3d(geometries, volume):
    converted = copy.deepcopy(geometries)

    # convert angles so that detector position is not the center point but the edge point
    # TODO: I should use decorator patterns here on geometries
    # shift to the center of the geom
    for ngeom, geom in zip(converted, geometries):
        dx = -(volume.extent_min[0] + volume.extent_max[0]) / 2
        dy = -(volume.extent_min[1] + volume.extent_max[1]) / 2
        dz = -(volume.extent_min[2] + volume.extent_max[2]) / 2
        ngeom.tube_position[:] = ngeom.tube_position + [dx, dy, dz]
        ngeom.detector_position[:] = ngeom.detector_position + [dx, dy, dz]

    scale = np.array(volume.voxel_size)
    print(f"scale {scale}")
    for ngeom in converted:
        # detector pixels have to be scaled first, because # detector.width and detector.height need to be scaled accordingly
        horiz_pixel_vector = (ngeom.u * ngeom.detector.pixel_width) / scale
        new_pixel_width = np.linalg.norm(horiz_pixel_vector)
        new_u_unit = horiz_pixel_vector / new_pixel_width
        ngeom.detector.pixel_width = new_pixel_width
        ngeom.u = new_u_unit

        vert_pixel_vector = (ngeom.v * ngeom.detector.pixel_height) / scale
        new_pixel_height = np.linalg.norm(vert_pixel_vector)
        new_v_unit = vert_pixel_vector / new_pixel_height
        ngeom.detector.pixel_height = new_pixel_height
        ngeom.v = new_v_unit

        ngeom.tube_position[:] = ngeom.tube_position[:] / scale
        ngeom.detector_position[:] = ngeom.detector_position[:] / scale

    for ngeom in converted:
        # TODO: check if this scaling way also is the case for FanBP
        ngeom.detector_position[:] = (
            ngeom.detector_position
            - ngeom.u * ngeom.detector.width / 2
            - ngeom.v * ngeom.detector.height / 2)

    return converted


def _compat_swap(geometries, volume=None, flip_uv=False):
    for geom in geometries:
        _swp = lambda x: np.array([x[2], x[1], x[0]])
        if flip_uv:
            geom.u[:], geom.v[:] = _swp(geom.v[:]), _swp(geom.u[:])
            geom.detector.pixel_width, geom.detector.pixel_height = \
                geom.detector.pixel_height, geom.detector.pixel_width
        else:
            geom.v[:], geom.u[:] = _swp(geom.v[:]), _swp(geom.u[:])

        geom.tube_position[:] = _swp(geom.tube_position[:])
        geom.detector_position[:] = _swp(geom.detector_position[:])

    if volume is not None:
        volume2_data = cp.transpose(volume.data, [2, 1, 0])
        volume2_extent_min = cp.flip(volume.extent_min)
        volume2_extent_max = cp.flip(volume.extent_max)
        volume2 = astrapy.Volume(volume2_data,
                                 volume2_extent_min,
                                 volume2_extent_max)
        return volume2
    else:
        return None


class ConeProjection(Kernel):
    ANGLES_PER_BLOCK = 4
    BLOCK_SLICES = 4
    COLS_PER_BLOCK = 32
    ROWS_PER_BLOCK = 32
    MAX_ANGLES = 1024

    def __init__(self, path: str = "../kernels/3d/cone_fp.cu"):
        super().__init__(path)
        names = ['cone_fp<DIR_X>', 'cone_fp<DIR_Y>', 'cone_fp<DIR_Z>']
        self.module = load_jinja2_module(
            path,
            name_expressions=names,
            max_angles=self.MAX_ANGLES,
            columns_per_block=self.COLS_PER_BLOCK,
            rows_per_block=self.ROWS_PER_BLOCK,
            block_slices=self.BLOCK_SLICES,
            angles_per_block=self.ANGLES_PER_BLOCK)
        self.cone_fp = [self.module.get_function(name) for name in names]

    def __call__(
        self,
        volume: astrapy.Volume,
        sino: astrapy.Sinogram,
        geometry: list[astrapy.Static3DGeometry],
        rays_per_pixel: int = 1
    ) -> astrapy.Sinogram:
        """Forward projection with fan geometry."""

        if isinstance(sino.data, np.ndarray):
            # TODO: silently replacing the object may be a bit unexpected,
            #   rather raise an error?
            sino.data = cp.asarray(sino.data, dtype=self.FLOAT_DTYPE)
        elif isinstance(sino.data, cp.ndarray):
            if sino.data.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`projections` must be a ndarray.")

        if sino.ndim != 3:
            raise ValueError("`projections` must have exactly 3 dimensions.")

        if isinstance(volume.data, np.ndarray):
            volume.data = cp.asarray(volume.data, dtype=self.FLOAT_DTYPE)
        elif isinstance(volume.data, cp.ndarray):
            if volume.data.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`volume` must be a Numpy or CuPy ndarray.")

        if volume.ndim != 3:
            raise ValueError("`volume` must have exactly 3 dimensions.")

        if not volume.has_isotropic_voxels:
            # TODO: this should work right?
            raise NotImplementedError(
                f"`{self.__class__.__name__}` is not tested with anisotropic voxels yet.")

        volume_texture = _cupy_copy_to_texture(volume.data)
        geometries = _prep_geom3d(geometry, volume)
        # _compat_swap(geometries)  # TODO
        volume = _compat_swap(geometries, volume, flip_uv=False)  # TODO

        # chunk arrays in blocks of MAX_ANGLES
        # TODO: is the MAX_ANGLES constraint still necessary
        for angle_start in range(0, len(geometries), self.MAX_ANGLES):
            angle_end = min(angle_start + self.MAX_ANGLES, len(geometries))
            self._call_chunk(volume_texture,
                             volume.data.shape,
                             sino.data[angle_start:angle_end],
                             sino.data.shape[0],
                             geometries[angle_start:angle_end],
                             rays_per_pixel=rays_per_pixel,
                             voxel_size=volume.voxel_size)

        return sino

    def _call_chunk(
        self,
        volume_texture: txt.TextureObject,
        volume_shape: tuple,
        projections: cp.ndarray,
        nr_projs: int,
        geoms: list,
        rays_per_pixel: int,
        voxel_size: tuple,
    ):
        """
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
        # C++ angleCount, note this differs from dims.iProjAngles
        # which corresponds to the total number of projection angles in the
        # geometry, while angleCount corresponds to the number of angles in
        # the chunk.
        nr_geoms = len(geoms)
        assert projections.shape[0] == nr_geoms

        eps = .0001  # TODO(Adriaan): think about the precision of this factor
        if (abs(voxel_size[0] / voxel_size[1] - 1.0) > eps or
            abs(voxel_size[0] / voxel_size[2] - 1.0) > eps):
            output_scale_x = [(voxel_size[1] / voxel_size[0]) ** 2,
                              (voxel_size[2] / voxel_size[0]) ** 2,
                              voxel_size[0] * voxel_size[0]]
            output_scale_y = [(voxel_size[0] / voxel_size[1]) ** 2,
                              (voxel_size[2] / voxel_size[1]) ** 2,
                              voxel_size[0] * voxel_size[1]]
            output_scale_z = [(voxel_size[0] / voxel_size[2]) ** 2,
                              (voxel_size[1] / voxel_size[2]) ** 2,
                              voxel_size[0] * voxel_size[2]]
            output_scale = [output_scale_x, output_scale_y, output_scale_z]
        else:  # cube is the same for all directions, thus * 3
            output_scale = [[1., 1., voxel_size[0]]] * 3

        srcsX = np.array([g.tube_position[0] for g in geoms])
        _cupy_copy_to_constant(self.module, 'srcsX', srcsX)
        srcsY = np.array([g.tube_position[1] for g in geoms])
        _cupy_copy_to_constant(self.module, 'srcsY', srcsY)
        srcsZ = np.array([g.tube_position[2] for g in geoms])
        _cupy_copy_to_constant(self.module, 'srcsZ', srcsZ)
        detsSX = np.array([g.detector_position[0] for g in geoms])
        _cupy_copy_to_constant(self.module, 'detsSX', detsSX)
        detsSY = np.array([g.detector_position[1] for g in geoms])
        _cupy_copy_to_constant(self.module, 'detsSY', detsSY)
        detsSZ = np.array([g.detector_position[2] for g in geoms])
        _cupy_copy_to_constant(self.module, 'detsSZ', detsSZ)
        detsUX = np.array([g.u[0] * g.detector.pixel_width for g in geoms])
        _cupy_copy_to_constant(self.module, 'detsUX', detsUX)
        detsUY = np.array([g.u[1] * g.detector.pixel_width for g in geoms])
        _cupy_copy_to_constant(self.module, 'detsUY', detsUY)
        detsUZ = np.array([g.u[2] * g.detector.pixel_width for g in geoms])
        _cupy_copy_to_constant(self.module, 'detsUZ', detsUZ)
        detsVX = np.array([g.v[0] * g.detector.pixel_height for g in geoms])
        _cupy_copy_to_constant(self.module, 'detsVX', detsVX)
        detsVY = np.array([g.v[1] * g.detector.pixel_height for g in geoms])
        _cupy_copy_to_constant(self.module, 'detsVY', detsVY)
        detsVZ = np.array([g.v[2] * g.detector.pixel_height for g in geoms])
        _cupy_copy_to_constant(self.module, 'detsVZ', detsVZ)

        # TODO(Adriaan): explicitly formalize that multiple detectors is
        #   problematic atm. Maybe consider more advanced geometries where
        #   the detector is shared.
        #   Note that multiple detectors is technically not a problem, they
        #   just render different u's and v's, and if they factually have
        #   less rows and columns, the computation is silently doing
        #   something that is similar to supersampling
        assert np.all(
            [g.detector.rows == geoms[0].detector.rows] for g in geoms)
        rows = geoms[0].detector.rows
        assert np.all(
            [g.detector.cols == geoms[0].detector.cols] for g in geoms)
        cols = geoms[0].detector.cols

        blocks_u = (cols + self.COLS_PER_BLOCK - 1) // self.COLS_PER_BLOCK
        blocks_v = (rows + self.ROWS_PER_BLOCK - 1) // self.ROWS_PER_BLOCK

        def _launch(angle_start, stop_angle, direction):
            blocks_projs = (stop_angle - angle_start
                            + self.ANGLES_PER_BLOCK - 1) // self.ANGLES_PER_BLOCK
            # TODO(ASTRA): check if we can't immediately destroy
            #              the stream after use
            # TODO(Adriaan):
            #   Now using `with stream.Stream()` but I should check
            #   if/how to run tun this loop with async streams (or
            #   are streams automatically async in CuPy?) in order to maximize
            #   performance. This could be a bottleneck atm, really
            #   need to check.
            with cp.cuda.stream.Stream():
                for i in range(0, volume_shape[direction], self.BLOCK_SLICES):
                    if rays_per_pixel != 1:
                        raise NotImplementedError(
                            "Detector supersampling is currently not supported.")

                    # TODO(Adriaan): why we are not using a 3D grid here?
                    self.cone_fp[direction](
                        (blocks_u * blocks_v, blocks_projs),
                        (self.COLS_PER_BLOCK, self.ANGLES_PER_BLOCK),
                        (volume_texture,
                         projections,
                         i,  # startSlice
                         angle_start,  # startAngle
                         stop_angle,  # endAngle
                         *volume_shape,
                         nr_projs,  # total number
                         cols,
                         rows,
                         cp.float32(output_scale[direction][0]),
                         cp.float32(output_scale[direction][1]),
                         cp.float32(output_scale[direction][2])))

        # Run over all angles, grouping them into groups of the same
        # orientation (roughly horizontal vs. roughly vertical).
        # Start a stream of grids for each such group.
        prev_stop_angle = 0
        prev_direction = None
        for stop_angle in range(nr_geoms + 1):
            # keep looping until the direction changes or we reach the end
            if not stop_angle == nr_geoms:
                direction = ConeProjection._calc_direction(geoms[stop_angle])
                if direction == prev_direction:
                    continue

            if prev_direction is not None:
                assert not prev_stop_angle == stop_angle
                # direction changed: launch kernels for angles with prev_dir
                _launch(prev_stop_angle, stop_angle, prev_direction)

            prev_direction = direction
            prev_stop_angle = stop_angle

        cp.cuda.Device().synchronize()

    @staticmethod
    def _calc_direction(g: AstraStatic3DGeometry) -> int:
        # TODO(Adriaan): its a bit weird that we converted to the corner, and
        #   now have to convert back. Make both part positions part of the geom?
        det_pos = (g.detector_position
                   + g.u * g.detector.width / 2
                   + g.v * g.detector.height / 2)
        d = np.abs(g.tube_position - det_pos)

        if d[0] >= d[1] and d[0] >= d[2]:
            direction = 0
        elif d[1] >= d[0] and d[1] >= d[2]:
            direction = 1
        else:
            direction = 2

        return direction


class ConeBackprojection(Kernel):
    ANGLES_PER_BLOCK = 32
    # last dim is not number of threads, but volume slab thickness
    VOL_BLOCK = (16, 32, 6)
    MAX_ANGLES = 1024

    def __init__(self, path: str = "../kernels/3d/cone_bp.cu"):
        super().__init__(path)

        self.module = load_jinja2_module(path,
                                         name_expressions=('cone_bp<true>',
                                                           'cone_bp<false>'),
                                         vol_block_x=self.VOL_BLOCK[0],
                                         vol_block_y=self.VOL_BLOCK[1],
                                         vol_block_z=self.VOL_BLOCK[2],
                                         angles_per_block=self.ANGLES_PER_BLOCK,
                                         max_angles=self.MAX_ANGLES)
        self.ops = {True: self.module.get_function("cone_bp<true>"),
                    False: self.module.get_function("cone_bp<false>")}

    def __call__(self,
                 volume: astrapy.Volume,
                 sino: astrapy.Sinogram,
                 geometry: list[astrapy.Static3DGeometry],
                 fdk_weighting: bool = False
                 ) -> astrapy.Volume:
        """Backprojection with fan geometry."""

        if isinstance(sino.data, np.ndarray):
            sino.data = cp.asarray(sino.data, dtype=self.FLOAT_DTYPE)
        elif isinstance(sino.data, cp.ndarray):
            if sino.data.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`sino` must be a ndarray.")

        if sino.data.ndim != 3:
            raise ValueError("`sino` must have exactly 3 dimensions.")

        if isinstance(volume.data, np.ndarray):
            volume.data = cp.asarray(volume.data, dtype=self.FLOAT_DTYPE)
        elif isinstance(volume.data, cp.ndarray):
            if volume.data.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`volume` must be a Numpy or CuPy ndarray.")

        # TODO: test, and also apply to other kernels
        assert volume.data.flags.c_contiguous is True, \
            f"`{self.__class__.__name__}` is not tested without C-contiguous data."

        if volume.ndim != 3:
            raise ValueError("`volume` must have exactly 3 dimensions.")

        if not volume.has_isotropic_voxels:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` is not tested with anisotropic voxels yet.")

        geometries = _prep_geom3d(geometry, volume)
        volume = _compat_swap(geometries, volume, flip_uv=True)  # TODO
        proj_texture = _cupy_copy_to_texture(sino.data)

        output_scale = 1.  # params.fOutputScale seems to be ~1. by default
        if fdk_weighting:  # ASTRA: NB: assuming cube voxels here
            output_scale /= volume.voxel_size[0]  # TODO
            raise NotImplementedError('Anisotropic voxels are not tested.')
        else:
            output_scale *= volume.voxel_volume

        # TODO(Adriaan): I think ceil() should do here (ande everywhere else)
        #  as well, or some array partitioning function
        blocks = [(s + b - 1) // b for s, b in
                  zip(volume.data.shape, self.VOL_BLOCK)]

        for angle_start in range(0, len(geometries), self.MAX_ANGLES):
            angle_end = min(angle_start + self.MAX_ANGLES,
                            len(geometries))
            # precompute kernel parameters
            geom_params = [ConeBackprojection._geom2params(
                geom, fdk_weighting, volume.voxel_size)
                for geom in geometries[angle_start:angle_end]]
            assert (len(geom_params) <= self.MAX_ANGLES)
            # TODO(Adriaan): was it not better to have all params grouped?
            #  for performance, AoS vs SoA
            params_list = [[*nU.to_list(), *nV.to_list(), *d.to_list()]
                           for (nU, nV, d) in geom_params]
            params = np.array(params_list).flatten().astype(np.float)
            assert not np.isnan(params).any()

            with cp.cuda.stream.Stream():
                _cupy_copy_to_constant(self.module, 'params', params)
                for a in range(0, angle_end, self.ANGLES_PER_BLOCK):
                    self.ops[fdk_weighting](
                        (blocks[0] * blocks[1], blocks[2]),
                        (self.VOL_BLOCK[0], self.VOL_BLOCK[1]),
                        (proj_texture,
                         volume.data,
                         a,
                         len(params_list),
                         *volume.data.shape,
                         cp.float32(output_scale)))

                cp.cuda.Device().synchronize()

        output_scale /= float(volume.voxel_volume)
        output_scale *= float(sino.pixel_volume)

        # TODO(Adriaan): ODL would be calling
        #    # Fix scaling to weight by pixel/voxel size
        #    out *= astra_cuda_bp_scaling_factor(
        #        self.proj_space, self.vol_space, self.geometry
        #    )
        #  here, adding an additional factor.

        return volume

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
        d = geom.detector_position

        if not fdk_weighting:
            cr = np.cross(u, v)
            # NB(ASTRA): for cross(u,v) we invert the volume scaling (for the voxel
            # size normalization) to get the proper dimensions for
            # the scaling of the adjoint
            cr[0] *= voxel_size[1] * voxel_size[2]
            cr[1] *= voxel_size[0] * voxel_size[2]
            cr[2] *= voxel_size[0] * voxel_size[1]
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
        den = _cuda_float4(w=scale * np.linalg.det([u, v, s]),
                           # = 1.0 for FDK
                           x=-scale * _det3x(u, v),
                           y=-scale * _det3y(u, v),
                           z=-scale * _det3z(u, v))
        return numU, numV, den
