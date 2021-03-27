from abc import ABC, abstractmethod
from pathlib import Path

import cupy as cp
import cupy.cuda.texture as txt
import jinja2
import numpy as np
from cupy.core.raw import RawModule
from cupy.cuda.runtime import (
    cudaAddressModeBorder, cudaChannelFormatKindFloat, cudaFilterModeLinear,
    cudaReadModeElementType, cudaResourceTypeArray)


def _cupy_copy_to_texture(obj) -> txt.TextureObject:
    """Creates a single-channel 2D/3D texture object of type float"""
    assert obj.ndim in [2, 3]

    # We're using an CUDA array resource type, which I think makes a copy,
    # but has more efficient access, compared to linear memory. I can imagine
    # that, when memory is an issue, it would be more performant to prefer
    # linear memory for projections.
    # TODO: it looks as if my definition of width and height are reversed,
    #  do I need Fortran order? Or is this maybe an inconsistency in
    #  the CUDA/Cupy interface?
    # TODO: test in 3D
    channel_desc = txt.ChannelFormatDescriptor(
        32, 0, 0, 0, cudaChannelFormatKindFloat)
    cuda_array = txt.CUDAarray(channel_desc, *reversed(obj.shape))
    cuda_array.copy_from(obj)

    resource_desc = txt.ResourceDescriptor(
        cudaResourceTypeArray, cuArr=cuda_array)
    texture_desc = txt.TextureDescriptor(
        [cudaAddressModeBorder] * obj.ndim,
        cudaFilterModeLinear,  # filter modebase = {NoneType} None
        cudaReadModeElementType)
    return txt.TextureObject(resource_desc, texture_desc)


def _cupy_copy_to_constant(
    module: RawModule,
    name: str,
    array,
    dtype=np.float32):
    """Copy array to constant memory
    Inspired by: https://github.com/cupy/cupy/issues/1703
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
