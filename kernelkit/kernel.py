import importlib.resources
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import resources
from typing import Sequence

import cupy as cp
import cupy.cuda.texture as txt
import jinja2
import numpy as np
from cupy.cuda.runtime import (
    cudaAddressModeBorder,
    cudaChannelFormatKindFloat,
    cudaFilterModeLinear,
    cudaReadModeElementType,
    cudaResourceTypeArray,
    cudaResourceTypePitch2D,
)

from kernelkit.data import ispitched

_texture_desc_2d = txt.TextureDescriptor(
    [cudaAddressModeBorder] * 2,  # array.ndim,
    cudaFilterModeLinear,  # filter modebase = {NoneType} None
    cudaReadModeElementType,
)
_texture_desc_3d = txt.TextureDescriptor(
    [cudaAddressModeBorder] * 3,  # array.ndim,
    cudaFilterModeLinear,  # filter modebase = {NoneType} None
    cudaReadModeElementType,
)
_channel_desc = txt.ChannelFormatDescriptor(32, 0, 0, 0, cudaChannelFormatKindFloat)


class KernelNotCompiledException(Exception):
    """Exception raised when a kernel is called before it has been compiled."""

    pass


class KernelMemoryUnpreparedException(Exception):
    """Raised when a kernel is called before memory has been prepared."""

    pass


def copy_to_texture(
    array: cp.ndarray, texture_type: str = "array", layered: bool = False
) -> txt.TextureObject:
    """Creates a single-channel 2D/3D texture object of type float
    from a CuPy array.

    Parameters
    ----------
    array : cupy.ndarray
        Array to be used in the texture. Must be pitched when `type` is
        'pitch2d'.
    texture_type : str, optional
        Type of texture to be created. Can be 'array', 'array2d', or 'pitch2d'.
        Defaults to 'array'. An 'array' texture may be faster, but requires
        the array to be copied to a CUDA array first, which increases memory
        usage.
    layered : bool, optional
        Whether to create a layered texture. Defaults to False. Only valid
        for 3D texture.

    Returns
    -------
    cupy.cuda.texture.TextureObject
        Texture object.
    """
    if not (isinstance(array, cp.ndarray) or isinstance(array, np.ndarray)):
        raise TypeError(f"Expected a NumPy or CuPy array, got {type(array)} instead.")

    if not array.dtype == cp.float32:
        raise ValueError(f"Array must be of type float32, got {array.dtype} instead.")

    if texture_type.lower() == "array" or texture_type.lower() == "array2d":
        if not array.flags["C_CONTIGUOUS"]:
            raise ValueError("Array must be C-contiguous.")

        assert (
            array.ndim == 2
            and texture_type.lower() == "array2d"
            or array.ndim == 3
            and texture_type.lower() == "array"
        ), f"Type '{texture_type}' is not compatible with array of ndim {array.ndim}."
        assert layered and array.ndim == 3 or not layered
        # TODO: CUDAarray memory allocation is not performed by CuPy memory
        #       management, and may cause a memory overflow.
        cuda_array = txt.CUDAarray(
            _channel_desc, *reversed(array.shape), 1 if layered else 0
        )
        cuda_array.copy_from(array, cp.cuda.get_current_stream())  # asynchronous copy
        resource_desc = txt.ResourceDescriptor(cudaResourceTypeArray, cuArr=cuda_array)
        return txt.TextureObject(resource_desc, _texture_desc_3d)
    elif texture_type.lower() == "pitch2d":
        assert array.ndim == 2
        array_base = array.base if array.base is not None else array
        assert array_base.ndim == 2
        if not array_base.flags["C_CONTIGUOUS"]:
            raise ValueError("Base array must be C-contiguous.")

        if not ispitched(array_base):
            raise ValueError(
                "Array data `array.base` needs to have pitched "
                "dimensions. Use `aspitched(array)`."
            )
        resource_desc = txt.ResourceDescriptor(
            cudaResourceTypePitch2D,
            arr=array,
            chDesc=_channel_desc,
            width=array_base.shape[1],
            height=array_base.shape[0],
            pitchInBytes=array_base.shape[1] * array.dtype.itemsize,
        )
        return txt.TextureObject(resource_desc, _texture_desc_2d)
    else:
        raise ValueError(f"Type `{texture_type}` not understood.")


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
    assert array.flags["C_CONTIGUOUS"]
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


class BaseKernel(ABC):
    """Abstract base class for CUDA kernels"""

    @abstractmethod
    def __init__(self, cuda_source: str):
        """Initialize the kernel"""

        # we cannot parse the source immediately because some Jinja2 template
        # arguments may only be known at compile time.
        self._cuda_source = cuda_source

    @abstractmethod
    def __call__(self, *args, **kwargs) -> type(None):
        # note we don't want kernels to return anything!
        # they work in-place and this would make batching more difficult
        pass

    @property
    def cuda_source(self) -> str:
        """Returns the CUDA source code of the kernel"""
        return self._cuda_source

    def _compile(
        self,
        name_expressions: Sequence[str],
        template_kwargs: dict = None
    ) -> cp.RawModule:
        """Renders Jinja2 template and imports kernel in CuPy

        Parameters
        ----------
        name_expressions : dict
            Dictionary of C++ functions to be compiled
        template_kwargs : dict
            Dictionary of template arguments to be rendered in the source code
            using Jinja2
        """
        template_kwargs = template_kwargs or {}
        if len(template_kwargs) == 0:
            print(f"Compiling kernel {self.__class__.__name__}:"
                  f" {', '.join(name_expressions)}...")
        else:
            print(
                f"Compiling kernel {self.__class__.__name__}:"
                f" {', '.join(name_expressions)}"
                f" with arguments {template_kwargs}"
                "..."
            )
        self.__compiled_template_kwargs = template_kwargs
        code = jinja2.Template(
            self.cuda_source, undefined=jinja2.StrictUndefined
        ).render(**template_kwargs)
        self._module = cp.RawModule(
            code=code,
            # --std is required for name expressions
            # -line-info for debugging
            options=("--std=c++17",),  # TODO: c++17 is allowed from CUDA 12
            name_expressions=name_expressions,
        )
        # TODO(Adriaan): add `jittify=False` when compilation is slow?
        return self._module

    @property
    def _compiled_template_kwargs(self):
        try:
            return self.__compiled_template_kwargs
        except AttributeError:
            raise KernelNotCompiledException(
                f"Cannot query the compiled template arguments of kernel "
                f"{self.__class__.__name__} before compilation.")

    @property
    def is_compiled(self):
        return hasattr(self, "_module")
