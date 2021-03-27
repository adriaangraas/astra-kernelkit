from typing import Any, Sequence, Sized

import numpy as np
from tqdm import tqdm

from astrapy import kernels


def fp(
    volume: Any,
    geometry: Any,
    projections_cpu: Sized = None,
    volume_extent_min: Sequence = None,
    volume_extent_max: Sequence = None,
    chunk_size: int = 100,
    **kwargs
):
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
    kernel = kernels.ConeProjection()

    if projections_cpu is None:
        # TODO: allocate in memory-friendly back-end (e.g. file-based)
        projections_cpu = []
        for g in geometry:
            d = np.zeros((g.detector.cols, g.detector.rows))
            projections_cpu.append(d)

    executor = kernels.chunk_coneprojection(
        kernel,
        volume=volume,
        volume_extent_min=volume_extent_min,
        volume_extent_max=volume_extent_max,
        chunk_size=chunk_size,
        projections_cpu=projections_cpu,
        geometries=geometry,
        **kwargs)

    for _ in tqdm(executor):
        pass

    return projections_cpu


def bp(
    projections: np.ndarray,
    geometry: Any,
    volume_shape: Sequence,
    volume_extent_min: Sequence,
    volume_extent_max: Sequence,
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

    executor = kernels.chunk_conebackprojection(
        kernels.ConeBackprojection(),
        projections_cpu=projections,
        geometries=geometry,
        volume_shape=volume_shape,
        volume_extent_min=volume_extent_min,
        volume_extent_max=volume_extent_max,
        chunk_size=chunk_size,
        filter=filter,
        **kwargs)

    for volume_gpu in tqdm(executor):
        pass

    return volume_gpu.get()
