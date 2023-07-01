from typing import Sequence, Tuple

import cupy as cp
import numpy as np

_PITCH = 32


def pitched_shape(array) -> Tuple[int, int, int]:
    bytes = (int(np.ceil(array.shape[-1] * array.dtype.itemsize / _PITCH))
             * _PITCH)
    items = bytes / array.dtype.itemsize
    assert items.is_integer()
    return *array.shape[:-1], int(items)


def ispitched(array) -> bool:
    arr = array.base if array.base is not None else array
    return pitched_shape(arr) == arr.shape


def aspitched(array, xp=None):
    """Pads array to pitched shape and returns a view in original shape.

    This function can also be used to transfer a non-pitched CPU array to a
    pitched GPU array.

    Parameters
    ----------
    xp : object Array module used for the new array in the pitched shape. If
    `None`, the module of the given array is used.
    """
    if xp is None:  # `xp` is the output array module
        xp = cp.get_array_module(array)

    if ispitched(array) and cp.get_array_module(array) == xp:
        return xp.asarray(array)

    pitched_array = xp.zeros(pitched_shape(array), dtype=array.dtype)
    # TODO(Adriaan): can the following be done without a copy?
    pitched_array[..., :array.shape[-1]] = xp.asarray(array[...])
    vw = pitched_array[..., :array.shape[-1]]
    assert vw.flags.owndata is False
    assert vw.base is pitched_array
    return vw
