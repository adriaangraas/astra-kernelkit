import collections
from typing import Any, Callable, List, Sequence, Sized

import cupy as cp
import numpy as np
from tqdm import tqdm

from astrapy.geom.proj import ProjectionGeometry
from astrapy.geom.vol import VolumeGeometry
from astrapy.kernel import Kernel
from astrapy.kernels.cone_bp import ConeBackprojection
from astrapy.operator import ConebeamTransform
from astrapy.processing import preweight, filter as filt
from astrapy.projector import ConeProjector, ConeBackprojector


def fp(volume: Any, projection_geometry: Any, volume_geometry: Any,
       out: Sized = None):
    op = ConebeamTransform(projection_geometry, volume_geometry)
    vol = cp.asarray(volume, dtype=cp.float32)
    result = op(vol, out=out)
    return result.get() if out is None else out


def bp(
    projections: Any,
    projection_geometry: List[ProjectionGeometry],
    volume_geometry: VolumeGeometry,
    filter: Any = None,
    preproc_fn: Callable = None,
    kernel: Kernel = None,
    out: cp.ndarray = None,
    projection_axes: Sequence[int] = (0, 1, 2),
    **kwargs):
    if out is None:
        out = cp.zeros(volume_geometry.shape, cp.float32)
    else:
        out = cp.asarray(out, dtype=cp.float32)

    projections = cp.asarray(projections, dtype=cp.float32)

    if preproc_fn is not None:
        preproc_fn(projections, projection_geometry)
    if filter is not None:
        if projection_axes != (0, 1, 2):
            raise NotImplementedError()
        preweight(projections, projection_geometry)
        filt(projections, filter=filter)

    kwargs.update(projection_axes=projection_axes)
    ptor = ConeBackprojector(
        ConeBackprojection(**kwargs) if kernel is None else kernel,
        texture_type='layered'
    )
    ptor.projection_geometry = projection_geometry
    ptor.volume_geometry = volume_geometry
    ptor.projections = projections
    ptor.volume = out
    ptor()
    return out.get()


def fdk(
    projections: Any,
    projection_geometry: Any,
    volume_geometry: VolumeGeometry,
    filter: Any = 'ramlak',
    preproc_fn: Callable = None,
    **kwargs):
    """Feldkamp-Davis-Kress algorithm"""

    return bp(projections,
              projection_geometry,
              volume_geometry,
              filter=filter,  # just bp with a filter
              preproc_fn=preproc_fn,
              **kwargs)


def sirt(
    projections: np.ndarray,
    projection_geometry: Any,
    volume_geometry: VolumeGeometry,
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

    """
    if len(projections) != len(projection_geometry):
        raise ValueError("Number of projections does not match number of"
                         " geometries.")
    if algo == 'cpu':
        xp_proj = np
    elif algo == 'gpu':
        xp_proj = cp
    else:
        raise ValueError

    xp_vol = cp

    # prevent copying if already in GPU memory, otherwise copy to GPU
    y = [xp_proj.array(p, copy=False) for p in projections]
    x = xp_vol.zeros(volume_geometry.shape, dtype)  # output volume
    x_tmp = xp_vol.ones_like(x)

    fptor = ConeProjector()
    fptor.projection_geometry = projection_geometry
    fptor.volume_geometry = volume_geometry

    bptor = ConeBackprojector(texture_type='pitch2d')
    bptor.projection_geometry = projection_geometry
    bptor.volume_geometry = volume_geometry

    def A(x, out=None):
        if out is None:
            out = xp_proj.zeros((len(projection_geometry),
                                 projection_geometry[0].detector.rows,
                                 projection_geometry[0].detector.cols), dtype)
        fptor.volume = x
        fptor.projections = out
        return fptor()

    def A_T(y, out=None):
        if out is None:
            out = xp_vol.zeros(x.shape, dtype=cp.float32)
        bptor.projections = y
        bptor.volume = out
        bptor()
        return out

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
        # Note, a choice made here is to keep `y_tmp` in GPU memory. This
        # requires more GPU memory, but is faster than the alternative.
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
