import copy
from typing import Sequence

import cupy as cp
import numpy as np


def voxel_size(volume_shape: Sequence,
               extent_min: Sequence,
               extent_max: Sequence) -> np.ndarray:
    """The physical size of a voxel."""
    n = np.array(volume_shape)
    dists = np.array(extent_max) - np.array(extent_min)
    shp = list(dists / n)
    return np.array(shp)


def has_isotropic_voxels(volume_shape: Sequence,
                         extent_min: Sequence,
                         extent_max: Sequence) -> bool:
    """Check if a voxel has the same length in each direction."""
    vox_size = voxel_size(volume_shape, extent_min, extent_max)
    return np.all(np.array(vox_size) == vox_size[0])


def voxel_volume(volume_shape: Sequence,
                 extent_min: Sequence,
                 extent_max: Sequence) -> float:
    vox_size = voxel_size(volume_shape, extent_min, extent_max)
    return float(np.prod(vox_size))


# class Sinogram:
#     """Wrapper for measured projection data for some times/angles
#
#     The wrapper
#      - adds some physicals dimensions that are required for good reconstructions;
#      - provides Numpy access to the data via `data`:
#      - allows the caller to obtain the data without memory transfer from the GPU, with `data_cupy`.
#     """
#
#     def __init__(self, data=None, axis_order: Sequence = None):
#         """Initialize a Sinogram data array
#
#         :param data:
#         :param axis_order: [0,1,2]
#         """
#         if not (isinstance(data, np.ndarray) or isinstance(data, cp.ndarray)):
#             raise TypeError("`data` has to be a Numpy or Cupy ndarray.")
#
#         if axis_order != [0, 1, 2]:
#             # TODO(Adriaan): check if correct
#             pass
#         else:
#             axis_order = tuple(range(data.ndim))
#
#         self._data = data
#         self.axis_order = axis_order
#
#     @property
#     def data_numpy(self) -> np.ndarray:
#         """Guarantees a Numpy array
#         Use if the caller expects Numpy or needs explicit CPU operations.
#         Might invoke a data transfer."""
#         if isinstance(self._data, cp.ndarray):
#             return cp.asnumpy(self._data)
#
#         return self._data
#
#     @property
#     def data(self):
#         """Guarantees an array implementing __array_function__
#         Returns either a Numpy or Cupy implementation, depending on the current internal state.
#         Does not induce a data transfer between GPU/CPU.
#         Use if you don't care where the array lives at the moment."""
#         return self._data
#
#     @data.setter
#     def data(self, value):
#         if not (isinstance(value, np.ndarray) or isinstance(value, cp.ndarray)):
#             raise TypeError("`data` has to be a ndarray.")
#
#         self._data = value
#
#     @property
#     def data_cupy(self) -> cp.ndarray:
#         """Guarantees a Cupy implementation
#         Use if you plan on explicit GPU or Cupy operations.
#         Might invoke a data transfer."""
#         return self._data
#
#     @property
#     def ndim(self) -> int:
#         """The dimensions of the data array."""
#         return self.data.ndim


def empty_gpu(shape, dtype=None, order=None):
    """An implementation-unaware array creation helper.
    Helps the user to write Cupy-free code.
    TODO: evaluate if it is realistic to expect that a Cupy-unaware user
    """
    return cp.empty(shape, dtype, order)


def zeros_gpu(shape, dtype=None, order=None):
    """An implementation-unaware array creation helper.
    Helps the user to write Cupy-free code.
    TODO: evaluate if it is realistic to expect that a Cupy-unaware user
    """
    return cp.zeros(shape, dtype, order)
