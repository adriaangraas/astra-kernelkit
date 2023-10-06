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
    cudaResourceTypePitch2D)

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
                    layered: bool = False) -> txt.TextureObject:
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
    layered : bool, optional
        Whether to create a layered texture. Defaults to False. Only valid
        for 3D textures.

    Returns
    -------
    cupy.cuda.texture.TextureObject
        Texture object.
    """
    if type.lower() == 'array' or type.lower() == 'array2d':
        assert (array.ndim == 2 and type.lower() == 'array2d'
                or array.ndim == 3 and type.lower() == 'array'), (
            f"Type '{type}' is not compatible with array of ndim {array.ndim}."
        )
        assert layered and array.ndim == 3 or not layered
        # TODO: CUDAarray memory allocation is not performed by CuPy memory
        #       management, and may cause a memory overflow.
        cuda_array = txt.CUDAarray(_channel_desc, *reversed(array.shape),
                                   1 if layered else 0)
        cuda_array.copy_from(array, cp.cuda.get_current_stream())
        resource_desc = txt.ResourceDescriptor(
            cudaResourceTypeArray, cuArr=cuda_array)
        return txt.TextureObject(resource_desc, _texture_desc_3d)
    elif type.lower() == 'pitch2d':
        assert array.ndim == 2
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
    p = module.get_global(name)
    # if cp.get_array_module(array) == np:
    #     array = np.squeeze(array)
    #     assert array.flags['C_CONTIGUOUS']
    #     assert array.dtype == np.float32
    #     p.copy_from_async(array.ctypes.data_as(ctypes.c_void_p), array.nbytes)
    # else:
    p.copy_from_async(array.base.data, array.base.nbytes)


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

    @abstractmethod
    def __call__(self, *args, **kwargs) -> type(None):
        # note we don't want kernels to return anything!
        # they work in-place and this would make batching more difficult
        pass

    @property
    def cuda_source(self) -> str:
        """Returns the CUDA source code of the kernel"""
        return self._cuda_source

    def _compile(self, name_expressions: Sequence[str], template_kwargs: dict) -> cp.RawModule:
        """Renders Jinja2 template and imports kernel in CuPy

        Parameters
        ----------
        name_expressions : dict
            Dictionary of C++ functions to be compiled
        template_kwargs : dict
            Dictionary of template arguments to be rendered in the source code
            using Jinja2
        """
        print(f"Compiling kernel {self.__class__.__name__}"
              f" with arguments {template_kwargs}"
              f"...")
        self._compiled_template_kwargs = template_kwargs
        code = jinja2.Template(
            self.cuda_source,
            undefined=jinja2.StrictUndefined).render(**template_kwargs)
        self._module = cp.RawModule(
            code=code,
            # --std is required for name expressions
            # -line-info for debugging
            options=('--std=c++11',),  # TODO: c++17 is allowed from CUDA 12
            name_expressions=name_expressions)
        # TODO(Adriaan): add `jittify=False` when compilation is slow?
        return self._module

    @property
    def is_compiled(self):
        return hasattr(self, '_module')