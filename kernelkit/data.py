"""Data manipulation utilities."""
import cupy as cp
import numpy as np

_CUDA_PITCH = 32  # the default CUDA pitch in bytes


def pitched_shape(array) -> tuple[int, int, int]:
    """Returns a suggestion for a pitched shape of an array, typically slightly
    larger than the original shape.

    Parameters
    ----------
    array : array-like
        The array to get the pitched shape of.

    Returns
    -------
    tuple
        The pitched shape of the array.
    """
    bytes = (
        int(np.ceil(array.shape[-1] * array.dtype.itemsize / _CUDA_PITCH)) * _CUDA_PITCH
    )
    items = bytes / array.dtype.itemsize
    assert items.is_integer()
    return *array.shape[:-1], int(items)


def ispitched(array) -> bool:
    """Returns `True` if the array is pitched, `False` otherwise.
    If the array is a view, the base array is checked.

    Parameters
    ----------
    array : array-like
        The array to check.

    Returns
    -------
    bool
        `True` if the array is pitched, `False` otherwise.
    """
    arr = array.base if array.base is not None else array
    return pitched_shape(arr) == arr.shape


def aspitched(array, xp=None):
    """Pads array to pitched shape and returns a view in original shape.

    This function can also be used to transfer a non-pitched CPU array to a
    pitched GPU array, by setting `xp` to `cp`.

    Parameters
    ----------
    array : array-like
        The array to pad to pitched shape.
    xp : array module, optional
        Array module used for the new array in the pitched shape. If
        `None`, the module of the given array is used.

    Returns
    -------
    array-like
        A view of the array in the original shape.
    """
    if xp is None:  # `xp` is the output array module
        xp = cp.get_array_module(array)

    if ispitched(array) and cp.get_array_module(array) == xp:
        return xp.asarray(array)

    pitched_array = xp.zeros(pitched_shape(array), dtype=array.dtype)
    # TODO(Adriaan): can the following be done without invoking a copy?
    pitched_array[..., : array.shape[-1]] = xp.asarray(array[...])
    vw = pitched_array[..., : array.shape[-1]]
    assert vw.flags.owndata is False
    assert vw.base is pitched_array
    return vw


def pitched_like(array, xp=None, fill=None):
    """Returns an empty array of same dimensions and same pitched base.

    Parameters
    ----------
    array : array-like
        The pitched array to get the pitched shape of.
    xp : array module, optional
        Array module used for the new array in the pitched shape. If
        `None`, the module of the given array is used.
    fill : scalar, optional
        Value to fill the array with. When `None`, the array is not filled.
        Defaults to `None`.

    Notes
    -----
    The purpose of this function is to avoid a copy when creating a new array
    with the same shape and pitched base as another array.
    """
    if xp is None: # `xp` is the output array module
        xp = cp.get_array_module(array)

    if not ispitched(array):
        raise ValueError("Input array needs to be pitched.")

    # override shape, and don't take the base shape
    # the input base could be much larger (e.g., (angles, row, columns)) while
    # the desired view requires a smaller base (e.g., (row, columns))
    base = xp.empty_like(array.base,
                         shape=pitched_shape(array))
    view = base[..., :array.shape[-1]]  # return the non-pitched view
    if fill is not None:
        view.fill(fill)
    return view