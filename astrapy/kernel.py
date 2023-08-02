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
    [cudaAddressModeBorder] * 3,  # array.ndim,
    cudaFilterModeLinear,  # filter modebase = {NoneType} None
    cudaReadModeElementType)
_channel_desc = txt.ChannelFormatDescriptor(
    32, 0, 0, 0, cudaChannelFormatKindFloat)


def copy_to_texture(array: cp.ndarray,
                    type: str = 'array',
                    layered=False) -> txt.TextureObject:
    """Creates a single-channel 2D/3D texture object of type float
    from a CuPy array.

    Parameters
    ----------
    array : cupy.ndarray
        Array to be copied to texture. Must be pitched when `type` is
        'pitch2d'.
    type : str, optional
        Type of texture to be created. Can be 'array' or 'pitch2d'.
        Defaults to 'array'. An 'array' texture may be faster, but requires
        the array to be copied to a CUDA array first, which increases memory
        usage.

    Returns
    -------
    cupy.cuda.texture.TextureObject
        Texture object.
    """
    if type.lower() == 'array' or type.lower():
        assert array.ndim in (2, 3)
        assert layered and array.ndim == 3 or not layered
        # TODO: for some reason CUDAarray's cause memory overflow, maybe
        #  `q` doesn't get cleaned up as it is still associated with the
        #  resource descriptor.. Try manually cleaning up `q`, maybe by
        #  force-deallocating or deleting it from the descriptor.
        cuda_array = txt.CUDAarray(_channel_desc, *reversed(array.shape),
                                   1 if layered else 0)
        cuda_array.copy_from(array, cp.cuda.get_current_stream())
        resource_desc = txt.ResourceDescriptor(
            cudaResourceTypeArray, cuArr=cuda_array)
        return txt.TextureObject(resource_desc, _texture_desc_3d)
    elif type.lower() == 'pitch2d':
        assert array.ndim in (2,)
        array_base = array.base if array.base is not None else array
        assert array_base.ndim == 2
        if not ap.ispitched(array_base):
            raise ValueError(
                "Array data `array.base` needs to have pitched "
                "dimensions. Use `aspitched(array)`.")
        resource_desc = txt.ResourceDescriptor(
            cudaResourceTypePitch2D, arr=array, chDesc=_channel_desc,
            width=array_base.shape[1], height=array_base.shape[0],
            pitchInBytes=array_base.shape[1] * array.dtype.itemsize)
        return txt.TextureObject(resource_desc, _texture_desc_2d)
    else:
        raise ValueError(f"Type `{type}` not understood.")


def texture_shape(obj: txt.TextureObject) -> tuple:
    """Shape of the array in the texture

    This does *not* return the pitched shape, but the original shape of the
    array, where possible. See `_copy_to_texture`.

    Parameters
    ----------
    obj : txt.TextureObject
        Texture object

    Returns
    -------
    tuple
        Shape of the array in the texture
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

    Parameters
    ----------
    module : cp.RawModule
        CUDA module
    name : str
        Name of the symbol to copy to (e.g. constant memory)
    array : np.ndarray
        Array to copy to the symbol
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
    """Abstract base class for CUDA kernels"""
    FLOAT_DTYPE = cp.float32
    SUPPORTED_DTYPES = [cp.float32]

    @abstractmethod
    def __init__(self, resource: str, package='astrapy.cuda'):
        """Initialize the kernel

        Parameters
        ----------
        resource : str
            Name of the resource file containing the kernel source
        package : str
            Name of the package where the resource file is located
        """
        # we cannot load the source immediately because some template
        # arguments may only be known at call time.
        self._cuda_source = resources.read_text(package=package,
                                                resource=resource)
        self.__compilation_cache = {}
        self.__compilation_times = 0

    @abstractmethod
    def __call__(self, *args, **kwargs) -> type(None):
        # note we don't want kernels to return anything!
        # they work in-place and this would make batching more difficult
        pass

    @property
    def cuda_source(self) -> str:
        """Returns the CUDA source code of the kernel"""
        return self._cuda_source

    def __compile(self,
                  name_expressions,
                  template_kwargs) -> cp.RawModule:
        """Renders Jinja2 template and imports kernel in CuPy

        Parameters
        ----------
        name_expressions : dict
            Dictionary of C++ functions to be compiled
        template_kwargs : dict
            Dictionary of template arguments to be rendered in the source code
            using Jinja2"""
        print(f"Compiling kernel {self.__class__.__name__}"
              # f" with arguments {template_kwargs}"
              f"...")
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
        """Compiles the kernel and caches the result

        Parameters
        ----------
        names : Sequence[str]
            Names of the C++ functions to be compiled
        template_kwargs : dict
            Dictionary of template arguments to be rendered in the source code
            using Jinja2

        Notes
        -----
        Technically, the kernel is only compiled when a function is retrieved
        from it."""
        h_names = hash(frozenset(names))
        h_kwargs = hash(frozenset(template_kwargs.items()))
        if not h_names in self.__compilation_cache:
            self.__compilation_cache[h_names] = {}

        if not h_kwargs in self.__compilation_cache[h_names]:
            self.__compilation_times += 1
            self.__compilation_cache[h_names][h_kwargs] = \
                self.__compile(names, template_kwargs)

        if self.__compilation_times > 5:
            warnings.warn(
                f"Module `{self.__class__.__name__}` has been recompiled "
                f"5 times, consider passing limits.")

        return self.__compilation_cache[h_names][h_kwargs]
