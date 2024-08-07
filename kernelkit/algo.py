from typing import Any, Callable, Sequence

import cupy as cp
import cupyx as cpx
import numpy as np
from tqdm import tqdm

from kernelkit.data import aspitched, pitched_like
from kernelkit.geom.proj import Beam, ProjectionGeometry
from kernelkit.geom.vol import VolumeGeometry
from kernelkit.kernel import BaseKernel
from kernelkit.kernels.cone_bp import VoxelDrivenConeBP
from kernelkit.operator import XrayTransform
from kernelkit.processing import preweight, filter as filt
from kernelkit.projector import ForwardProjector, BackProjector


def fp(
    volume: Any,
    projection_geometry: list[ProjectionGeometry],
    volume_geometry: VolumeGeometry,
    out=None,
):
    """X-ray forward projection.

    Parameters
    ----------
    volume: array-like
        The volume to project.
    projection_geometry: list[ProjectionGeometry]
        The projection geometries.
    volume_geometry: VolumeGeometry
        The volume geometry.
    out: array-like, optional
        The projection array to write the result to. If not given, a new
        array is allocated on the CPU in page-locked memory and returned.

    Returns
    -------
    out : ndarray
        The forward projection of the volume.
    """
    op = XrayTransform(projection_geometry, volume_geometry)
    vol = cp.asarray(volume, dtype=cp.float32)
    result = op(vol, out=out)
    if out is None:
        out = cpx.zeros_like_pinned(result, dtype=cp.float32)
    result.get(out=out)
    return out


def bp(
    projections: Any,
    projection_geometry: list[ProjectionGeometry],
    volume_geometry: VolumeGeometry,
    filter: Any = None,
    preproc_fn: Callable = None,
    kernel: BaseKernel = None,
    out=None,
    projection_axes: Sequence[int] = (0, 1, 2),
    **kwargs,
):
    """X-ray backprojection.

    Parameters
    ----------
    projection_axes : Sequence[int], optional
        The axes of the projections. Default is (0, 1, 2).
    projections : array-like
        Either a 2D list of projections, or a 3D array of projections.
    projection_geometry : list[ProjectionGeometry]
        The projection geometries.
    volume_geometry : VolumeGeometry
        The volume geometry.
    filter : Any
        The filter to apply to the projections before backprojection.
    preproc_fn : Callable, optional
        A function to apply to the projections before backprojection,
        especially suited for pre-processing on the GPU. Default is None.
    out : array-like, optional
        The volume array to write the result to. If not given, a new array is
        allocated on the CPU in page-locked memory and returned.

    Returns
    -------
    out : ndarray
        The volume that was backprojected into.

    Warnings
    --------
    The API of this function will likely change in some future because of
    the kernel passing mechanism.
    """
    if out is None:
        out_gpu = cp.zeros(volume_geometry.shape, cp.float32)
    else:
        out_gpu = cp.asarray(out, dtype=cp.float32)

    projections = cp.asarray(projections, dtype=cp.float32)

    if preproc_fn is not None:
        preproc_fn(projections, projection_geometry)
    if filter is not None:
        if projection_axes != (0, 1, 2):
            raise NotImplementedError()
        preweight(projections, projection_geometry)
        filt(projections, filter=filter)

    # TODO(Adriaan): rewrite using the `XrayTransform` class
    kwargs.update(projection_axes=projection_axes)
    kwargs.update(beam=projection_geometry[0].beam)
    ptor = BackProjector(
        VoxelDrivenConeBP(**kwargs) if kernel is None else kernel,
        texture_type="layered",
    )
    ptor.projection_geometry = projection_geometry
    ptor.volume_geometry = volume_geometry
    ptor.projections = projections
    ptor.volume = out_gpu
    ptor()

    if out is None or isinstance(out, cp.ndarray):
        out_cpu = cpx.zeros_like_pinned(out_gpu, dtype=cp.float32)
        return out_gpu.get(out=out_cpu)
    elif isinstance(out, np.ndarray):
        return out_gpu.get(out=out)


def fbp(
    projections: Any,
    projection_geometry: list[ProjectionGeometry],
    volume_geometry: VolumeGeometry,
    filter: Any = "ramlak",
    **bp_kwargs,
):
    """Filtered backprojection algorithm for parallel-beam geometry.

    Parameters
    ----------
    projections : array-like
        The projections to reconstruct from.
    projection_geometry : list[ProjectionGeometry]
        The projection geometries.
    volume_geometry : VolumeGeometry
        The volume geometry.
    filter : Any, optional
        The filter to apply to the projections before backprojection. Default
        is 'ramlak'. See `kernelkit.processing.filter` for more information.
    bp_kwargs : dict, optional
        Keyword arguments to pass to the `bp` function. See
        `kernelkit.algo.bp` for more information.

    See Also
    --------
    kernelkit.algo.bp : Backprojection function.
    """
    raise NotImplementedError(
        "FBP is not implemented yet. Use a conebeam geometry with the `bp` function, or"
        " file an issue."
    )

    if not "kernel" in bp_kwargs:
        bp_kwargs.update(kernel=VoxelDrivenConeBP())
    return bp(
        projections, projection_geometry, volume_geometry, filter=filter, **bp_kwargs
    )


def fdk(
    projections: Any,
    projection_geometry: Any,
    volume_geometry: VolumeGeometry,
    filter: Any = "ramlak",
    **bp_kwargs,
):
    """Feldkamp-Davis-Kress algorithm for conebeam geometry.

    Parameters
    ----------
    projections : array-like
        The projections to reconstruct from.
    projection_geometry : list[ProjectionGeometry]
        The projection geometries.
    volume_geometry : VolumeGeometry
        The volume geometry.
    filter : Any, optional
        The filter to apply to the projections before backprojection. Default
        is 'ramlak'. See `kernelkit.processing.filter` for more information.
    bp_kwargs : dict, optional
        Keyword arguments to pass to the `bp` function. See
        `kernelkit.algo.bp` for more information.

    See Also
    --------
    kernelkit.algo.bp : Backprojection function.

    Notes
    -----
    The FDK algorithm is a special case of the `bp` function, with a
    `VoxelDrivenConeBP` kernel and a filter. See the article by Feldkamp et
    al. [1]_ for more information.

    References
    ----------
    .. [1] Feldkamp, L. A., Davis, L. C., & Kress, J. W. (1984). Practical
        cone-beam algorithm. Journal of the Optical Society of America A,
        1(6), 612. https://doi.org/10.1364/JOSAA.1.000612

    """
    for p in projection_geometry:
        if p.beam != Beam.CONE:
            raise NotImplementedError(
                "Only cone beam geometry is supported at the moment."
            )
    if not "kernel" in bp_kwargs:
        bp_kwargs.update(kernel=VoxelDrivenConeBP())
    return bp(
        projections,
        projection_geometry,
        volume_geometry,
        filter=filter,  # just bp with a filter
        **bp_kwargs,
    )


def sirt(
    projections: np.ndarray,
    projection_geometry: Any,
    volume_geometry: VolumeGeometry,
    iters: int = 100,
    verbose: bool = True,
    mask: Any = None,
    min_constraint: float = None,
    max_constraint: float = None,
    return_gpu: bool = False,
    callback: callable = None,
    residual_every: int = 10,
):
    """Simultaneous Iterative Reconstruction Technique.

    Parameters
    ----------
    projections : array-like
        The projections to reconstruct from. Must follow axes dimensions
         (0, 1, 2), i.e., (projections, rows, cols). Other conventions are
         possible but not implemented here.
    projection_geometry : list[ProjectionGeometry]
        The projection geometries.
    volume_geometry : VolumeGeometry
        The volume geometry.
    iters : int, optional
        The number of iterations to perform. Default is 100.
    verbose : bool, optional
        Whether to print a progress bar. Default is True.
    mask : array-like, optional
        A mask to apply to the volume after each iteration. By default, no
        mask is applied.
    min_constraint : float, optional
        The minimum value to clip the volume to after each iteration. By
        default, no clipping is applied.
    max_constraint : float, optional
        The maximum value to clip the volume to after each iteration. By
        default, no clipping is applied.
    return_gpu : bool, optional
        Whether to return the volume on the GPU. Default is False.
    callback : callable, optional
        A function to call after each iteration. The function should take
        three arguments: the iteration number, the volume, and the residual.
        Default is None.
    residual_every : int, optional
        How often to compute the MSE residual for display. Default is every 10
        iterations. Note that computing the residual is expensive and may slow
        down the reconstruction. It is best to set this value conservatively.
    """
    if len(projections) != len(projection_geometry):
        raise ValueError("Number of projections does not match number of geometries.")
    xp_proj = cp
    xp_vol = cp

    # prevent copying if already in GPU memory, otherwise copy to GPU
    # y = [xp_proj.array(p, copy=False) for p in projections]
    y = [aspitched(p, xp=xp_proj) for p in projections]
    x = xp_vol.zeros(volume_geometry.shape, cp.float32)  # output volume
    x_tmp = xp_vol.ones_like(x)

    fptor = ForwardProjector()
    fptor.projection_geometry = projection_geometry
    fptor.volume_geometry = volume_geometry

    def A(x, out=None):
        if out is None:
            out = xp_proj.zeros(
                (
                    len(projection_geometry),
                    projection_geometry[0].detector.rows,
                    projection_geometry[0].detector.cols,
                ),
                cp.float32,
            )
        fptor.volume = x
        fptor.projections = out
        fptor()
        return out

    bptor = BackProjector(texture_type="pitch2d")
    bptor.projection_geometry = projection_geometry
    bptor.volume_geometry = volume_geometry

    def A_T(y, out=None):
        if out is None:
            out = xp_vol.zeros(x.shape, dtype=cp.float32)
        bptor.projections = y
        bptor.volume = out
        bptor()
        return out

    if mask is not None:
        mask = xp_vol.asarray(mask, cp.float32)

    # compute scaling matrix C
    # y_tmp = [aspitched(cp.ones_like(p)) for p in y]
    y_tmp = [pitched_like(p, xp_proj, fill=1.) for p in y]  # intermediate variable
    C = A_T(y_tmp)
    xp_vol.divide(1.0, C, out=C)  # TODO(Adriaan): there is no `where=` in CuPy
    C[C == xp_vol.infty] = 0.0
    cp.get_default_memory_pool().free_all_blocks()

    # compute scaling operator R
    R = A(x_tmp)
    for p in R:
        xp_proj.divide(1.0, p, out=p)  # TODO(Adriaan)
        p[p == xp_proj.infty] = 0.0
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

        if residual_every is not None:
            E = residual_every  # compute residual norm every `E`
            if i % E == 0:  # the following operation is expensive
                res_norm = str(sum([xp_proj.linalg.norm(p) for p in y_tmp]))
            desc = "SIRT [" + "*" * (i % E) + "." * (E - i % E) + "]"
            bar.set_description(f"{desc}: {res_norm})")
        else:
            bar.set_description(f"SIRT")

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
