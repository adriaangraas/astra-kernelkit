import copy
import warnings
from typing import Any, Callable, Sequence, Sized

import cupy as cp
import numpy as np
from tqdm import tqdm

from astrapy import kernels, process
from astrapy.data import aspitched, voxel_size
from astrapy.geom3d import shift
from astrapy.kernel import Kernel, _copy_to_texture


def suggest_volume_extent(geometry, object_position: Sequence = (0., 0., 0.)):
    # TODO(Adriaan): now using only one geometry.
    #    perhaps use polygon clipping on multiple geoms
    #   to find the intersection of volume areas.
    cg = copy.deepcopy(geometry)  # TODO
    shift(cg, -np.array(object_position))

    # assert that origin is on the source-detector line
    source_vec = cg.tube_position
    det_vec = cg.detector_position - cg.tube_position
    if not np.linalg.matrix_rank([source_vec, det_vec]) == 1:
        warnings.warn("Volume extents may not be suggested correctly when "
                      " the geometry does not project through the origin.")

    SOD = np.linalg.norm(source_vec)
    SDD = np.linalg.norm(det_vec)
    # using that pixel_width / SDD = voxel_width / SOD
    # => voxel_width = pixel_width / SOD * SDD
    width = SOD * cg.detector.width / SDD
    height = SOD * cg.detector.height / SDD
    return (np.array([-width / 2, -width / 2, -height / 2]),
            np.array([width / 2, width / 2, height / 2]))


def _voxel_size_from_wished_shape(wished_shape: Sequence,
                                  vol_ext_min: Sequence,
                                  vol_ext_max: Sequence):
    """Compute voxel size based on intended shape"""
    # convert scalars to arrays
    if np.all([s is None for s in wished_shape]):
        raise ValueError("Suggest at least one shape dimension.")

    if np.isscalar(wished_shape):
        wished_shape = [wished_shape] * 3
    wished_shape = np.array(wished_shape)
    vol_ext_min = np.array(vol_ext_min)
    vol_ext_max = np.array(vol_ext_max)

    # first find a voxel size given by the users wishes
    voxel_size = [None, None, None]
    for i in range(len(wished_shape)):
        if wished_shape[i] is None:
            continue

        voxel_size[i] = (vol_ext_max[i] - vol_ext_min[i]) / wished_shape[i]

    # if some dimensions are None, replicate the minimum voxel size
    voxel_size = np.array(voxel_size)
    voxel_size[voxel_size == None] = (
        np.min(voxel_size[np.where(voxel_size != None)]))
    return voxel_size.astype(np.float)


def _parse_vol_params(geometries, wished_shape,
                      volume_extent_min, volume_extent_max):
    vol_extents_given = (
        volume_extent_min is None and volume_extent_max is None)
    if vol_extents_given:
        volume_extent_min, volume_extent_max = suggest_volume_extent(
            geometries[0], (0., 0., 0))  # assume geometry is centered
    # compute voxel size from the users wished shape
    voxel_size = _voxel_size_from_wished_shape(
        wished_shape, volume_extent_min, volume_extent_max)
    # make volume slightly bigger to have isotropic voxels
    vol_size = np.array(volume_extent_max) - volume_extent_min
    actual_shape = np.ceil(vol_size / voxel_size).astype(np.int)
    # compute actual volume size by rounded shape, and thence ext min/max
    scaling_ratio = (actual_shape * voxel_size) / vol_size
    actual_ext_min = volume_extent_min * scaling_ratio
    actual_ext_max = volume_extent_max * scaling_ratio
    if not vol_extents_given:
        if np.any([scaling_ratio != 1.]):
            warnings.warn(f"Volume extents are scaled by a factor "
                          f"{scaling_ratio} and are now min, max = "
                          f"{volume_extent_min}, and {volume_extent_max} as "
                          f"to have isotropic voxels.")

    return actual_shape, actual_ext_min, actual_ext_max


def fp(
    volume: Any,
    geometry: Any,
    projections: Sized = None,
    volume_extent_min: Sequence = None,
    volume_extent_max: Sequence = None,
    chunk_size: int = 100,
    **kwargs
):
    """
    
    :type volume: object
        If `None` an `ndarray` is created on CPU, to save GPU memory.
        Then the result is transferred chunk-after-chunk to the CPU array.
        However, if a GPU array is given, no transfers are initiated.
    :type sinogram: object
        If `None` an `ndarray` is created on CPU, to save GPU memory.
        Then the result is transferred chunk-after-chunk to the CPU array.
        However, if a GPU array is given, no transfers are initiated.
    :type chunk_size: int
        If `None` all sinogram is processed at once. This might lead to a
        GPU memory overflow of some sort. When an integer is given, will
        upload and launch a kernel for each consecutive chunk of `sinogram`
        and `geometry`.
    """
    kernel = kernels.ConeProjection()

    if projections is None:
        # TODO: allocate in memory-friendly back-end (e.g. file-based)
        projections = []
        for g in geometry:
            d = np.zeros((g.detector.cols, g.detector.rows))
            projections.append(d)

    vol_shp, vol_ext_min, vol_ext_max = _parse_vol_params(
        geometry, volume.shape, volume_extent_min, volume_extent_max)
    executor = kernels.chunk_coneprojection(
        kernel,
        volume=volume,
        volume_extent_min=vol_ext_min,
        volume_extent_max=vol_ext_max,
        chunk_size=chunk_size,
        projections_cpu=projections,
        geometries=geometry,
        **kwargs)

    for _ in tqdm(executor):
        pass

    return projections


def bp(
    projections: np.ndarray,
    geometry: Any,
    volume_shape: Sequence,
    volume_extent_min: Sequence = None,
    volume_extent_max: Sequence = None,
    chunk_size: int = 1000,
    filter: str = 'ram_lak',
    **kwargs):
    """
    Executes `kernel`
    TODO: generate argument hints for editors for geometries that have to
    be passed through kwargs

    :type volume: object
        If `None` an `ndarray` is created on CPU, to save GPU memory.
        Then the result is transferred chunk-after-chunk to the CPU array.
        However, if a GPU array is given, no transfers are initiated.
    :type sinogram: object
        If `None` an `ndarray` is created on CPU, to save GPU memory.
        Then the result is transferred chunk-after-chunk to the CPU array.
        However, if a GPU array is given, no transfers are initiated.
    :type chunk_size: int
        If `None` all sinogram is processed at once. This might lead to a
        GPU memory overflow of some sort. When an integer is given, will
        upload and launch a kernel for each consecutive chunk of `sinogram`
        and `geometry`.

    """
    if len(projections) != len(geometry):
        raise ValueError

    if chunk_size is None:
        # TODO: allow None and auto infer good chunk size
        raise NotImplementedError()

    vol_shp, vol_ext_min, vol_ext_max = _parse_vol_params(
        geometry, volume_shape, volume_extent_min, volume_extent_max)

    executor = kernels.chunk_conebackprojection(
        kernels.ConeBackprojection(),
        projections_cpu=projections,
        geometries=geometry,
        volume_shape=vol_shp,
        volume_extent_min=vol_ext_min,
        volume_extent_max=vol_ext_max,
        chunk_size=chunk_size,
        filter=filter,
        **kwargs)

    for volume_gpu in tqdm(executor):
        pass

    return volume_gpu.get()

