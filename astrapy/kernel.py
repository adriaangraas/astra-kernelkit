import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence, Sized

import cupy as cp
import cupy.cuda.texture as txt
import jinja2
import numpy as np
from cupy.core.raw import RawModule
from cupy.cuda.runtime import cudaAddressModeBorder, \
    cudaChannelFormatKindFloat, cudaFilterModeLinear, cudaReadModeElementType, \
    cudaResourceTypeArray


def _cupy_copy_to_texture(obj) -> txt.TextureObject:
    """Creates a 2D / 3D texture of float format"""
    assert obj.ndim in [2, 3]

    # TODO(Adriaan): how do I choose these values?
    # TODO(Adriaan): do I have a precision choice?
    channel_desc = txt.ChannelFormatDescriptor(32, 0, 0, 0,
                                               cudaChannelFormatKindFloat)

    # We're using an CUDA array resource type, which I think makes a copy,
    # but has more efficient access, compared to linear memory. I can imagine
    # that, when memory is an issue, it would be more performant to prefer
    # linear memory for projections.
    # TODO: it looks as if my definition of width and height are reversed,
    #  do I need Fortran order? Or is this maybe an inconsistency in
    #  the CUDA/Cupy interface?
    # TODO: test in 3D
    texture_array = txt.CUDAarray(channel_desc, *reversed(obj.shape))
    texture_array.copy_from(obj)

    resource_desc = txt.ResourceDescriptor(
        cudaResourceTypeArray,
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


class Kernel(ABC):
    FLOAT_DTYPE = cp.float32
    SUPPORTED_DTYPES = [cp.float32]

    @abstractmethod
    def __init__(self, path: str, allow_recompilation=False):
        """
        Note: I don't want Kernel to do anything with global memory
        allocation. Managing global memory (with on/offloading, limits, CuPy
        pools, etc. is way too extensive to be handled within a kernel itself.

        :param path:
        :param allow_recompilation:
        """
        # we cannot load the source immediately because some template
        # arguments may only be known at runtime.
        self._cuda_source_path = Path(path)
        self._allow_recompilation = allow_recompilation
        self._nr_compilations = 0
        self._compilation_result = None

    @abstractmethod
    def __call__(self, *args, **kwargs) -> type(None):
        # note we don't want cuda to return anything!
        # they work in-place and this would make batching more difficult
        pass

    @property
    def cuda_source(self) -> Path:
        return self._cuda_source_path

    def load_module(self,
                    name_expressions=None,
                    **template_kwargs):
        """Renders Jinja2 template and imports kernel in CuPy"""
        code = jinja2.Template(
            self.cuda_source.read_text(),
            undefined=jinja2.StrictUndefined).render(**template_kwargs)
        return RawModule(
            code=code,
            # --std is required for name expressions
            # TODO: error on c++17, but NVRTC manual supports it?
            options=('--std=c++11',),
            name_expressions=name_expressions)

    @abstractmethod
    # def _compile(self, *args, **kwargs) -> tuple[RawModule, Any]:
    def _compile(self, *args, **kwargs) -> tuple:
        pass

    # def _get_compilation(self, *args, **kwargs) -> tuple[RawModule, Any]:
    def _get_compilation(self, *args, **kwargs):
        if self._compilation_result is None or self._allow_recompilation:
            if self._allow_recompilation:
                # I have to check if the arguments changed and recompilation
                # is necessary.
                raise NotImplementedError()

            self._compilation_result = self._compile(*args, **kwargs)
            assert self._compilation_result is not None

        return self._compilation_result


def _compat_swap_geoms(geometries):
    for geom in geometries:
        _swp = lambda x: np.array([x[2], x[1], x[0]])

        # if swap_uv:
        #     tmp = geom.v[:]
        #     geom.v[:] = geom.u[:]
        #     geom.u[:] = tmp
        #     tmp = geom.detector.pixel_width
        #     geom.detector.pixel_width = geom.detector.pixel_height
        #     geom.detector.pixel_height = tmp
        #     tmp = geom.detector.rows
        #     geom.detector.rows = geom.detector.cols
        #     geom.detector.cols = tmp

        geom.u[:] = _swp(geom.u[:])
        geom.v[:] = _swp(geom.v[:])
        geom.tube_position[:] = _swp(geom.tube_position[:])
        geom.detector_position[:] = _swp(geom.detector_position[:])


def _prep_geom3d(geometries,
                 volume_extent_min,
                 volume_extent_max,
                 volume_voxel_size):
    converted = copy.deepcopy(geometries)

    # convert angles so that detector position is not the center point but the edge point
    # TODO: I should use decorator patterns here on geometries
    # shift to the center of the geom
    dx = -(volume_extent_min[0] + volume_extent_max[0]) / 2
    dy = -(volume_extent_min[1] + volume_extent_max[1]) / 2
    dz = -(volume_extent_min[2] + volume_extent_max[2]) / 2
    for ngeom, geom in zip(converted, geometries):
        ngeom.tube_position[:] = ngeom.tube_position + [dx, dy, dz]
        ngeom.detector_position[:] = ngeom.detector_position + [dx, dy, dz]

    scale = np.array(volume_voxel_size)
    print(f"scale {1/scale}")
    for ngeom in converted:
        # detector pixels have to be scaled first, because
        # detector.width and detector.height need to be scaled accordingly
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

    return converted


