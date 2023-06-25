import collections
import copy
import warnings
from typing import Any, Callable, Sequence, Sized

import cupy as cp
import numpy as np
from tqdm import tqdm
import astrapy as ap
from astrapy.projector import ConeProjector, ConeBackprojector


def suggest_volume_extent(geometry, object_position: Sequence = (0., 0., 0.)):
    # TODO(Adriaan): now using only one geometry.
    #    perhaps use polygon clipping on multiple geoms
    #   to find the intersection of volume areas.
    cg = copy.deepcopy(geometry)  # TODO
    ap.shift_(cg, -np.array(object_position))

    # assert that origin is on the source-detector line
    source_vec = cg.source_position
    det_vec = cg.detector_position - cg.source_position
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


def vol_params(
    shp,
    ext_min,
    ext_max,
    vox_sz=None,
    geometries=None,
    verbose=True
):
    """Compute voxel size based on intended shape"""

    def _process_arg(arg):
        if np.isscalar(arg) or arg is None:
            arg = [arg] * 3

        return list(arg)

    shp = _process_arg(shp)
    ext_min = _process_arg(ext_min)
    ext_max = _process_arg(ext_max)
    vox_sz = _process_arg(vox_sz)

    def _resolve(n, xmin, xmax, sz, dim):
        """
        Resolving the equation, for a dimension `d`:
            nr_voxels (`n`) * voxel_size (`sz`) = volume extent (`xmax - xmin`)
        If two of these parameters are unknown, the third is resolved.
        """
        inferred = False
        if n is not None:
            if xmin is not None and xmax is not None:
                x = xmax - xmin
                if sz is not None:
                    assert np.allclose(n * sz, x), (
                        f"{n} voxels * {sz} voxel_size must equal extent"
                        f" in dim {dim}.")
                else:
                    sz = x / n  # resolved

                inferred = True
            elif xmin is None and xmax is None:
                if sz is not None:
                    xmin = -n * sz / 2
                    xmax = n * sz / 2
                    inferred = True
                else:
                    pass  # unresolved
            else:
                raise ValueError(
                    f"Volume extent in dim {dim} must be given with both max"
                    " and min value, or inferred automatically.")
        else:
            if xmin is not None and xmax is not None:
                if sz is not None:
                    x = xmax - xmin
                    n = x / sz
                    if not np.allclose(n, np.round(n)):
                        raise ValueError(
                            f"Number of voxels in dim {dim} was inferred to be"
                            f" {n}, which is not a rounded number. To have "
                            f" isotropic voxels, please give in larger"
                            f" volume size, or leave the volume extent out in"
                            f" dim {dim} to have it inferred automatically.")
                    n = int(np.round(n))
                    inferred = True
                else:
                    pass  # not inferred
            elif xmin is None and xmax is None:
                pass  # two unknowns: not inferred
            else:
                raise ValueError(
                    f"Volume extent in dim {dim} must be given with both max"
                    " and min value, or inferred automatically.")

        if inferred:
            if verbose:
                print("Computed volume parameters: ")
                print(f" - shape: {shp}")
                print(f" - extent min: {ext_min}")
                print(f" - extent max: {ext_max}")
                print(f" - voxel size: {vox_sz}")

            return (n, xmin, xmax, sz)

        return False

    resolved_dims = [False] * 3

    def _resolve_dims(dims, vol_ext_min, vol_ext_max):
        # first resolve any open dims
        for d in dims:
            if resolved_dims[d]:
                continue

            attempt = _resolve(shp[d], vol_ext_min[d], vol_ext_max[d],
                               vox_sz[d], d)
            if attempt:
                shp[d] = attempt[0]
                ext_min[d] = attempt[1]
                ext_max[d] = attempt[2]
                vox_sz[d] = attempt[3]
                resolved_dims[d] = True

    # try to resolve at least one dimension, so that we have a voxel size
    _resolve_dims(range(3), ext_min, ext_max)

    # if we failed to do that, throw in automatic geometry inference
    if not np.any(resolved_dims):
        if geometries is not None:
            sugg_ext_min, sugg_ext_max = suggest_volume_extent(
                geometries[0], (0., 0., 0))  # assume geometry is centered

            # try to put in a known geometry value, first x, then y, then z
            for d, resolved in enumerate(resolved_dims):
                if resolved:
                    continue

                try_ext_min = np.copy(ext_min)
                try_ext_max = np.copy(ext_max)
                try_ext_min[d] = sugg_ext_min[d]
                try_ext_max[d] = sugg_ext_max[d]
                _resolve_dims([d], try_ext_min, try_ext_max)

                if resolved_dims[d]:
                    break  # we're happy with one dim, as we'd like isotropy

    # at least one dimension should be resolved now
    if not np.any(resolved_dims):
        # the user probably didn't give enough
        raise ValueError("Not enough information is provided to infer a "
                         "voxel size, number of voxels and volume extent "
                         "of at least a single dimension. Consider "
                         "passing a geometry for automatic inference.")

    # replicate the minimum voxel size to other dimensions, if necessary
    vox_sz = np.array(vox_sz)
    vox_sz[vox_sz == None] = (np.min(vox_sz[np.where(vox_sz != None)]))
    vox_sz.astype(np.float32)
    # retry a resolve of the rest of the equations
    _resolve_dims(range(3), ext_min, ext_max)

    # replicate x, y dims, if necessary
    if resolved_dims[0] and not resolved_dims[1]:
        shp[1] = shp[0]
    if resolved_dims[1] and not resolved_dims[0]:
        shp[0] = shp[1]
    # retry a resolve of the rest of the equations
    _resolve_dims(range(2), ext_min, ext_max)

    # if any remaining, one dim has only a voxel size, and no nr_voxels or geom
    if not np.all(resolved_dims) and geometries is not None:
        # in that case, use the geom
        sugg_ext_min, sugg_ext_max = suggest_volume_extent(
            geometries[0], (0., 0., 0))  # assume geometry is centered

        # try to put in a known geometry value
        for d, resolved in enumerate(resolved_dims):
            if resolved:
                continue

            try_ext_min = np.copy(ext_min)
            try_ext_max = np.copy(ext_max)
            # round up geometry to have exact number of voxels
            extent = sugg_ext_max[d] - sugg_ext_min[d]
            nr_voxels_required = extent / vox_sz[d]
            # enlarge the geometry slightly, if necessary to have full voxels
            if np.allclose(np.round(nr_voxels_required), nr_voxels_required):
                nr_voxels_required = np.round(nr_voxels_required)
            else:
                nr_voxels_required = np.ceil(nr_voxels_required)
            try_ext_min[d] = - nr_voxels_required / 2 * vox_sz[d]
            try_ext_max[d] = nr_voxels_required / 2 * vox_sz[d]
            _resolve_dims([d], try_ext_min, try_ext_max)

    if not np.all(resolved_dims):
        raise RuntimeError("Could not resolve volume and voxel dimensions.")

    if verbose:
        print("Computed volume parameters: ")
        print(f" - shape: {shp}")
        print(f" - extent min: {ext_min}")
        print(f" - extent max: {ext_max}")
        print(f" - voxel size: {vox_sz}")
    return tuple(shp), tuple(ext_min), tuple(ext_max), tuple(vox_sz)


def fp(
    volume: Any,
    geometry: Any,
    volume_extent_min: Sequence = None,
    volume_extent_max: Sequence = None,
    volume_voxel_size: Sequence = None,
    out: Sized = None,
    kernel: ap.Kernel = None,
    **kwargs):
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
    _, vol_ext_min, vol_ext_max, _ = vol_params(
        volume.shape, volume_extent_min, volume_extent_max,
        volume_voxel_size, geometry)
    ptor = ap.ConeProjector(
        ap.ConeProjection() if kernel is None else kernel,
        volume_extent_min=np.asarray(vol_ext_min),
        volume_extent_max=np.asarray(vol_ext_max),
        **kwargs)
    ptor.volume = volume.astype(np.float32)
    ptor.geometry = geometry
    result = ptor(out)
    return result.get() if out is None else out


def bp(
    projections: Any,
    geometry: Any,
    volume_shape: Sequence,
    volume_extent_min: Sequence = None,
    volume_extent_max: Sequence = None,
    volume_voxel_size: Sequence = None,
    filter: Any = None,
    preproc_fn: Callable = None,
    verbose: bool = False,
    kernel: ap.Kernel = None,
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
    vol_shp, vol_ext_min, vol_ext_max, _ = vol_params(
        volume_shape, volume_extent_min, volume_extent_max,
        volume_voxel_size, geometry, verbose=verbose)
    ptor = ConeBackprojector(
        ap.ConeBackprojection() if kernel is None else kernel,
        volume_shape=np.asarray(vol_shp),
        volume_extent_min=np.asarray(vol_ext_min),
        volume_extent_max=np.asarray(vol_ext_max),
        filter=filter,
        preproc_fn=preproc_fn,
        verbose=verbose,
        **kwargs)
    ptor.geometry = geometry
    ptor.projections = projections
    result = ptor(out)
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
    """
    if len(projections) != len(geometry):
        raise ValueError("Number of projections does not match number of"
                         " geometries.")

    vol_shp, vol_ext_min, vol_ext_max, _ = vol_params(
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
    x = xp_vol.zeros(vol_shp, dtype)  # output volume
    x_tmp = xp_vol.ones_like(x)
    fptor = ConeProjector(ap.ConeProjection(), volume_extent_min,
                          volume_extent_max, verbose=verbose)
    fptor.geometry = geometry
    bptor = ConeBackprojector(ap.ConeBackprojection(), vol_shp,
                              vol_ext_min, vol_ext_max, verbose=verbose)
    bptor.geometry = geometry

    def A(x):
        fptor.volume = x
        return fptor()

    def A_T(y):
        bptor.projections = y
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
        y_tmp = A(x)  # forward project `x` into `y_tmp`
        # We need speed:
        # A(x, out=y_tmp)

        # compute residual in `y_tmp`, apply R
        for p_tmp, p, r in zip(y_tmp, y, R):
            p_tmp -= p  # residual
            p_tmp *= r

        E = 10  # compute residual norm every `E`
        if i % E == 0:  # this operation is expensive
            res_norm = str(sum([xp_proj.linalg.norm(p) for p in y_tmp]))
        desc = 'SIRT [' + '*' * (i % E) + '.' * (E - i % E) + ']'
        bar.set_description(f"{desc}: {res_norm})")

        # backproject residual into `x_tmp`, apply C
        x_tmp = A_T(y_tmp)
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
