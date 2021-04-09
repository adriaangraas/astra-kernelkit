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


def pitched_shape(array) -> tuple:
    assert array.flags.c_contiguous
    bytes = (int(np.ceil(array.shape[-1] * array.dtype.itemsize / 32)) * 32)
    items = bytes / array.dtype.itemsize
    assert items.is_integer()
    return *array.shape[:-1], int(items)


def ispitched(array) -> bool:
    arr = array.base if array.base is not None else array
    return pitched_shape(arr) == arr.shape


def aspitched(array):
    """Pads array to pitched shape and returns view in original shape."""
    assert array.ndim == 2
    if ispitched(array):
        return array

    xp = cp.get_array_module(array)
    pitched_array = xp.zeros(pitched_shape(array), dtype=array.dtype)
    pitched_array[:, :array.shape[1]] = array[...]
    vw = pitched_array[:, :array.shape[1]]
    assert vw.flags.owndata is False
    assert vw.base is pitched_array
    return vw
