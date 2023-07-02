import collections
from typing import Any, Callable, Sequence, Sized

import cupy as cp
import numpy as np
from tqdm import tqdm

from astrapy.operator import ConebeamTransform
from astrapy.kernel import Kernel
from astrapy.geom import resolve_volume_geometry
from astrapy.geom.vol import VolumeGeometry
from astrapy.projector import ConeProjector, ConeBackprojector
from astrapy.kernels.cone_fp import ConeProjection
from astrapy.kernels.cone_bp import ConeBackprojection


def fp(
    volume: Any,
    geometry: Any,
    volume_extent_min: Sequence = None,
    volume_extent_max: Sequence = None,
    volume_voxel_size: Sequence = None,
    out: Sized = None,
    kernel: Kernel = None,
    verbose: bool = False):
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
    # ptor = ConeProjector(ConeProjection() if kernel is None else kernel)
    # ptor.projection_geometry = geometry
    # ptor.volume = volume.astype(np.float32)
    volume_geometry = resolve_volume_geometry(
        volume.shape, volume_extent_min, volume_extent_max,
        volume_voxel_size, geometry, verbose=verbose)
    # ptor.projections = cp.zeros((len(geometry), geometry[0].detector.rows,
    #                                 geometry[0].detector.cols), cp.float32)
    # result = ptor()
    op = ConebeamTransform(geometry, volume_geometry)
    return result.get() if out is None else out


def bp(
    projections: Any,
    projection_geometry: Any,
    volume_shape: Sequence,
    volume_extent_min: Sequence = None,
    volume_extent_max: Sequence = None,
    volume_voxel_size: Sequence = None,
    filter: Any = None,
    preproc_fn: Callable = None,
    verbose: bool = False,
    kernel: Kernel = None,
    out: cp.ndarray = None,
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
    vol_geom = resolve_volume_geometry(
        shape=np.asarray(volume_shape),
        extent_min=np.asarray(volume_extent_min),
        extent_max=np.asarray(volume_extent_max),
        verbose=verbose)
    ptor = ConeBackprojector(
        ConeBackprojection() if kernel is None else kernel,
        filter=filter,
        preproc_fn=preproc_fn,
        verbose=verbose,
        **kwargs)
    volume = out if out is not None else cp.zeros(vol_geom.shape, cp.float32)
    ptor.projections = projections
    ptor.volume = volume
    ptor.projection_geometry = projection_geometry
    ptor.volume_geometry = vol_geom
    result = ptor()
    return result.get() if out is None else out


def fdk(
    projections: Any,
    geometry: Any,
    volume_shape: Sequence,
    volume_extent_min: Sequence = None,
    volume_extent_max: Sequence = None,
    volume_voxel_size: Sequence = None,
    filter: Any = 'ramlak',
    preproc_fn: Callable = None,
    **kwargs):
    """Feldkamp-Davis-Kress algorithm"""

    return bp(projections,
              geometry,
              volume_shape,
              volume_extent_min,
              volume_extent_max,
              volume_voxel_size,
              filter=filter,  # just bp with filter
              preproc_fn=preproc_fn,
              **kwargs)


def sirt(
    projections: np.ndarray,
    geometry: Any,
    volume_shape: Sequence,
    volume_extent_min: Sequence = None,
    volume_extent_max: Sequence = None,
    volume_voxel_size: Sequence = None,
    iters: int = 100,
    dtype=cp.float32,
    verbose: bool = True,
    mask: Any = None,
    proj_mask: Any = True,
    min_constraint: float = None,
    max_constraint: float = None,
    return_gpu: bool = False,
    algo='gpu',
    callback: callable = None):
    """Simulateneous Iterative Reconstruction Technique

    :param proj_mask: If `None` doesn't use one, unless `mask` is given,
        then it generates the appropriate one.
    :return:

    Parameters
    ----------
    volume_voxel_size
    volume_extent_max
    volume_extent_min
    geometry
    """
    if len(projections) != len(geometry):
        raise ValueError("Number of projections does not match number of"
                         " geometries.")

    vol_geom = resolve_volume_geometry(
        volume_shape, volume_extent_min, volume_extent_max,
        volume_voxel_size, geometry, verbose=verbose)

    if algo == 'cpu':
        xp_proj = np
    elif algo == 'gpu':
        xp_proj = cp
    else:
        raise ValueError

    xp_vol = cp

    # prevent copying if already in GPU memory, otherwise copy to GPU
    y = [xp_proj.array(p, copy=False) for p in projections]
    x = xp_vol.zeros(vol_geom.shape, dtype)  # output volume
    x_tmp = xp_vol.ones_like(x)
    fptor = ConeProjector()
    fptor.projection_geometry = geometry
    fptor.volume_geometry = vol_geom
    bptor = ConeBackprojector()
    bptor.projection_geometry = geometry
    bptor.volume_geometry = vol_geom

    def A(x, out=None):
        if out is None:
            out = xp_proj.zeros((len(geometry), geometry[0].detector.rows,
                                 geometry[0].detector.cols), dtype)
        fptor.volume = x
        fptor.projections = out
        return fptor()

    def A_T(y, out=None):
        if out is None:
            out = xp_vol.zeros(x.shape, dtype=cp.float32)
        bptor.projections = y
        bptor.volume = out
        return bptor()

    if mask is not None:
        mask = xp_vol.asarray(mask, dtype)
        if proj_mask is True:
            type = cp.int32 if xp_proj == cp else np.int32
            proj_mask = [(m > 0).astype(type) for m in A(mask)]

    if isinstance(proj_mask, collections.abc.Sequence):
        for y_i, m_i in zip(y, proj_mask):
            y_i *= m_i
        del proj_mask
        cp.get_default_memory_pool().free_all_blocks()

    # compute scaling matrix C
    # y_tmp = [aspitched(cp.ones_like(p)) for p in y]
    y_tmp = [xp_proj.ones_like(p) for p in y]  # intermediate variable
    C = A_T(y_tmp)
    xp_vol.divide(1., C, out=C)  # TODO(Adriaan): there is no `where=` in CuPy
    C[C == xp_vol.infty] = 0.
    cp.get_default_memory_pool().free_all_blocks()

    # compute scaling operator R
    R = A(x_tmp)
    for p in R:
        xp_proj.divide(1., p, out=p)  # TODO(Adriaan)
        p[p == xp_proj.infty] = 0.
    cp.get_default_memory_pool().free_all_blocks()

    bar = tqdm(range(iters), disable=not verbose, desc="SIRT")
    for i in bar:
        # Note, not using `out`: `y_tmp` is not recycled because it would
        # require `y_tmp` and its texture counterpart to be in memory at
        # the same time. We take the memory-friendly and slightly more GPU
        # intensive approach here.
        # y_tmp = A(x)  # forward project `x` into `y_tmp`
        A(x, out=y_tmp)

        # compute residual in `y_tmp`, apply R
        for p_tmp, p, r in zip(y_tmp, y, R):
            p_tmp -= p  # residual
            p_tmp *= r

        E = 10  # compute residual norm every `E`
        if i % E == 0:  # the following operation is expensive
            res_norm = str(sum([xp_proj.linalg.norm(p) for p in y_tmp]))
        desc = 'SIRT [' + '*' * (i % E) + '.' * (E - i % E) + ']'
        bar.set_description(f"{desc}: {res_norm})")

        # backproject residual into `x_tmp`, apply C
        A_T(y_tmp, out=x_tmp)
        x_tmp *= C
        x -= x_tmp  # update `x`
        if mask is not None:
            x *= mask
        if min_constraint is not None or max_constraint is not None:
            cp.clip(x, a_min=min_constraint, a_max=max_constraint, out=x)
        if callback is not None:
            callback(i, x, y_tmp)
        cp.get_default_memory_pool().free_all_blocks()

    if xp_vol == cp and not return_gpu:
        return x.get()
