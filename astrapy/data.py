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
                         extent_max: Sequence,
                         atol: float = 1e-8) -> bool:
    """Check if a voxel has the same length in each direction."""
    vox_size = voxel_size(volume_shape, extent_min, extent_max)
    return np.allclose(vox_size, vox_size[0], atol=atol)


def voxel_volume(volume_shape: Sequence,
                 extent_min: Sequence,
                 extent_max: Sequence) -> float:
    vox_size = voxel_size(volume_shape, extent_min, extent_max)
    return float(np.prod(vox_size))


def pitched_shape(array):
    assert array.ndim == 2
    # TODO: check if array is C-contiguous
    bytes = (int(np.ceil(array.shape[1] * array.dtype.itemsize / 32)) * 32)
    items = bytes / array.dtype.itemsize
    assert items.is_integer()
    return array.shape[0], int(items)


def ispitched(array) -> bool:
    return pitched_shape(array) == array.shape


def aspitched(array, pitched_view=True):
    assert array.ndim == 2
    if ispitched(array):
        return array

    xp = cp.get_array_module(array)
    pitched_array = xp.zeros(pitched_shape(array), dtype=xp.float32)
    pitched_array[:, :array.shape[1]] = array[...]

    if pitched_view:
        vw = pitched_array[:, :array.shape[1]]
        assert vw.flags.owndata is False
        assert vw.base is pitched_array
        return vw
    else:
        return array


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
