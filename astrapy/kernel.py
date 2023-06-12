import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import resources
from typing import Sequence

import astrapy as ap
import cupy as cp
import cupy.cuda.texture as txt
import jinja2
import numpy as np
from cupy.cuda.runtime import (
    cudaAddressModeBorder, cudaChannelFormatKindFloat, cudaFilterModeLinear,
    cudaReadModeElementType, cudaResourceTypeArray,
    cudaResourceTypePitch2D, cudaResourceTypeLinear)

_texture_desc_2d = txt.TextureDescriptor(
    [cudaAddressModeBorder] * 2,  # array.ndim,
    cudaFilterModeLinear,  # filter modebase = {NoneType} None
    cudaReadModeElementType)
_texture_desc_3d = txt.TextureDescriptor(
    [cudaAddressModeBorder] * 2,  # array.ndim,
    cudaFilterModeLinear,  # filter modebase = {NoneType} None
    cudaReadModeElementType)
_channel_desc = txt.ChannelFormatDescriptor(
    32, 0, 0, 0, cudaChannelFormatKindFloat)


def copy_to_texture(array, type: str = 'array') -> txt.TextureObject:
    """Creates a single-channel 2D/3D texture object of type float"""
    assert array.ndim in [2, 3]
    if type.lower() == 'array':
        # We're using an CUDA array resource type, which I think makes a copy,
        # but has more efficient access, compared to linear memory. I can imagine
        # that, when memory is an issue, it would be more performant to prefer
        # linear memory for projections.
        # TODO: for some reason CUDAarray's cause memory overflow, maybe
        #  `q` doesn't get cleaned up as it is still associated with the
        #  resource descriptor.. Try manually cleaning up `q`, maybe by
        #  force-deallocating or deleting it from the descriptor.
        cuda_array = txt.CUDAarray(_channel_desc, *reversed(array.shape))
        cuda_array.copy_from(array)  # async
        resource_desc = txt.ResourceDescriptor(
            cudaResourceTypeArray, cuArr=cuda_array)
    elif type.lower() == 'pitch2d' or type.lower() == 'linear':
        array_base = array.base if array.base is not None else array
        if type.lower() == 'pitch2d':
            if not ap.ispitched(array_base):
                raise ValueError(
                    "Array data `array.base` needs to have pitched "
                    "dimensions. Use `aspitched(array)`.")
            res_type = cudaResourceTypePitch2D
        else:
            res_type = cudaResourceTypeLinear

        # In `arr` we are putting a possible view object, so that the original
        # shape can be retrieved later using `_texture_shape`.
        assert array.base.ndim == 2
        resource_desc = txt.ResourceDescriptor(
            res_type, arr=array, chDesc=_channel_desc,
            width=array_base.shape[1], height=array_base.shape[0],
            pitchInBytes=array_base.shape[1] * array.dtype.itemsize)
    else:
        raise ValueError(f"`type` {type} not understood.")

    texture_desc = _texture_desc_2d if array.ndim == 2 else _texture_desc_3d
    return txt.TextureObject(resource_desc, texture_desc)


def texture_shape(obj: txt.TextureObject) -> tuple:
    """Shape of the array in the texture

    This does *not* return the pitched shape, but the original shape of the
    array, where possible. See `_copy_to_texture`.
    """
    rs = obj.ResDesc
    if rs.arr is not None:
        return rs.arr.shape  # resource may be pitched, but arr may be view
    elif rs.cuArr is not None:
        if rs.cuArr.ndim == 1:
            return rs.cuArr.width
        elif rs.cuArr.ndim == 2:
            return rs.cuArr.height, rs.cuArr.width
        elif rs.cuArr.ndim == 3:
            return rs.cuArr.depth, rs.cuArr.height, rs.cuArr.width

    raise ValueError("Texture Resource Descriptor not understood.")


def copy_to_symbol(module: cp.RawModule, name: str, array):
    """Copy array to address on GPU, e.g. constant memory

    Inspired by: https://github.com/cupy/cupy/issues/1703
    See also: https://docs.chainer.org/en/v1.3.1/_modules/cupy/creation/from_data.html
    """
    import ctypes
    array = np.squeeze(array)
    assert array.flags['C_CONTIGUOUS']
    assert array.dtype == np.float32
    p = module.get_global(name)
    # a_cpu = np.ascontiguousarray(np.squeeze(array), dtype=dtype)
    if cp.get_array_module(array) == np:
        p.copy_from_async(array.ctypes.data_as(ctypes.c_void_p), array.nbytes)
    else:
        p.copy_from_async(array.data, array.nbytes)


@dataclass(frozen=True, order=True)
class cuda_float4:
    """Helper to encode CUDA float4 in the right order"""
    x: int
    y: int
    z: int
    w: int

    def to_list(self):
        return [self.x, self.y, self.z, self.w]

    def __str__(self):
        return str(self.to_list())


class Kernel(ABC):
    FLOAT_DTYPE = cp.float32
    SUPPORTED_DTYPES = [cp.float32]

    @abstractmethod
    def __init__(self, resource: str, *args):
        """
        Note: Kernel should not do anything with global memory allocation.
        Managing global memory (with on/offloading, limits, CuPy
        pools, etc. is way too extensive to be handled within a kernel itself.

        :param path:
        :param allow_recompilation:
        """
        # we cannot load the source immediately because some template
        # arguments may only be known at runtime.
        self._cuda_source = resources.read_text('astrapy.cuda', resource)
        self.__compilation_cache = {}
        self.__compilation_times = 0

    @abstractmethod
    def __call__(self, *args, **kwargs) -> type(None):
        # note we don't want kernels to return anything!
        # they work in-place and this would make batching more difficult
        pass

    @property
    def cuda_source(self) -> str:
        return self._cuda_source

    def __compile(self,
                  name_expressions,
                  template_kwargs) -> cp.RawModule:
        """Renders Jinja2 template and imports kernel in CuPy"""
        print(f"Compiling kernel {self.__class__.__name__}...")
        code = jinja2.Template(
            self.cuda_source,
            undefined=jinja2.StrictUndefined).render(**template_kwargs)
        return cp.RawModule(
            code=code,
            # --std is required for name expressions
            # -line-info for debugging
            options=('--std=c++11',),  # TODO: c++17 is allowed from CUDA 12
            name_expressions=name_expressions)
        # TODO(Adriaan): add `jittify=False` when compilation is slow?

    def _compile(self,
                 names: Sequence[str],
                 template_kwargs: dict) -> cp.RawModule:
        h_names = hash(frozenset(names))
        h_kwargs = hash(frozenset(template_kwargs.items()))
        if not h_names in self.__compilation_cache:
            self.__compilation_cache[h_names] = {}

        if not h_kwargs in self.__compilation_cache[h_names]:
            self.__compilation_times += 1
            self.__compilation_cache[h_names][h_kwargs] = \
                self.__compile(names, template_kwargs)

        if self.__compilation_times > 5:
            # technically the kernel is only compiled when a function
            # is retrieved from it
            warnings.warn(
                f"Module `{self.__class__.__name__}` has been recompiled "
                f"5 times, consider passing limits.")

        return self.__compilation_cache[h_names][h_kwargs]
