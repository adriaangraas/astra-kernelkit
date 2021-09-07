import copy
import warnings
from typing import Any, Callable, Sequence, Sized

import cupy as cp
import numpy as np
from tqdm import tqdm

from astrapy import kernels, process
from astrapy.data import aspitched
from astrapy.geom import shift
from astrapy.kernel import _to_texture
from astrapy.kernels import ConeBackprojection, ConeProjection


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


def _parse_vol_params_from_extent(geometries, volume_shape,
                                  wished_extent_min, wished_extent_max):
    vol_extents_given = (
        wished_extent_min is not None and wished_extent_max is not None)
    if not vol_extents_given:
        wished_extent_min, wished_extent_max = suggest_volume_extent(
            geometries[0], (0., 0., 0))  # assume geometry is centered

    # compute voxel size from the users wished shape
    voxel_size = _voxel_size_from_wished_shape(
        volume_shape, wished_extent_min, wished_extent_max)

    # make volume bigger to have isotropic voxels
    # vol_size = np.array(wished_extent_max) - wished_extent_min
    # actual_shape = np.ceil(vol_size / voxel_size).astype(np.int)
    # actual_shape = np.ceil((vol_size / voxel_size) * np.max(voxel_size))
    # compute actual volume size by rounded shape, and thence ext min/max
    # scaling_ratio = (actual_shape * voxel_size) / vol_size
    actual_ext_min = wished_extent_min / voxel_size * np.max(voxel_size)
    actual_ext_max = wished_extent_max / voxel_size * np.max(voxel_size)
    if not vol_extents_given:
        if np.any([voxel_size != voxel_size[0]]):
            warnings.warn(f"Volume extents are scaled to accomodate to "
                          f" isotropic voxels and the given shape "
                          f"of {volume_shape}")

    return actual_ext_min, actual_ext_max


def _parse_vol_params_from_shape(geometries, wished_shape,
                                 volume_extent_min, volume_extent_max):
    vol_extents_given = (
        volume_extent_min is not None and volume_extent_max is not None)
    if not vol_extents_given:
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


def _coneprojection(
    kernel: ConeProjection,
    volume,
    volume_extent_min,
    volume_extent_max,
    chunk_size: int,
    geometries: list,
    projections_cpu: np.ndarray,
    dtype=cp.float32,
    **kwargs):
    """
    Allocates GPU memory for only `chunk_size` projection images, then
    repeats the kernel call into the same GPU memory.

    :param kernel:
    :param chunk_size:
    :param geometries:
    :param projections_cpu:
    :param kwargs:
    """
    assert chunk_size > 0
    volume_texture = _to_texture(volume)
    with cp.cuda.stream.Stream() as stream:
        for start_proj in tqdm(range(0, len(geometries), chunk_size),
                               desc="Forward projecting"):
            next_proj = min(start_proj + chunk_size, len(geometries))
            sub_projs = projections_cpu[start_proj:next_proj]
            projs_gpu = [cp.zeros(p.shape, dtype=dtype) for p in sub_projs]
            stream.synchronize()

            kernel(volume_texture,
                   volume_extent_min,
                   volume_extent_max,
                   geometries[start_proj:next_proj],
                   projs_gpu,
                   **kwargs)
            stream.synchronize()

            for cpu, gpu in zip(sub_projs, projs_gpu):
                # TODO: check performance is improved with async/pinned memory
                cpu[:] = gpu.get()

            yield


def _conebackprojection(
    kernel: ConeBackprojection,
    projections: list,
    geometries: list,
    volume_shape: Sequence,
    volume_extent_min: Sequence,
    volume_extent_max: Sequence,
    chunk_size: int = None,
    dtype=cp.float32,
    filter: Any = None,
    preproc_fn: Callable = None,
    **kwargs):
    """
    If one of the `projections` is on CPU, use `chunk_size` to upload, process,
    compute, and download projections in batches. Alternatively, if `projections`
    lives on the GPU, compute all as a whole.
    """

    def _preproc_to_texture(projs, geoms):
        if preproc_fn is not None:
            preproc_fn(projs)
            stream.synchronize()

        if filter is not None:
            process.preweight(projs, geoms)
            process.filter(projs, filter=filter)
            stream.synchronize()

        return _to_texture(projs)

    volume = cp.zeros(tuple(volume_shape), dtype=dtype)
    def _compute(projs_txt, geoms):
        kernel(
            projs_txt,
            geoms,
            volume,
            volume_extent_min,
            volume_extent_max)

    # run chunk-based algorithm if one or more projections are on CPU
    if np.any([isinstance(p, np.ndarray) for p in projections]):
        assert chunk_size > 0
        with cp.cuda.stream.Stream() as stream:
            for start in tqdm(range(0, len(geometries), chunk_size),
                              desc="Backprojecting"):
                end = min(start + chunk_size, len(geometries))
                sub_geoms = geometries[start:end]
                sub_projs = projections[start:end]
                sub_projs_gpu = cp.asarray(sub_projs)
                projs_txt = _preproc_to_texture(sub_projs_gpu, sub_geoms)
                stream.synchronize()
                _compute(projs_txt, sub_geoms)
                stream.synchronize()
                yield volume
    else:
        projs_txt = _preproc_to_texture(projections, geometries)
        _compute(projs_txt, geometries)

    return volume


def fp(
    volume: Any,
    geometry: Any,
    volume_extent_min: Sequence = None,
    volume_extent_max: Sequence = None,
    chunk_size: int = 100,
    out: Sized = None,
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

    if out is None:
        # TODO: allocate in memory-friendly back-end (e.g. file-based)
        out = []
        for g in geometry:
            d = np.zeros((g.detector.rows, g.detector.cols), dtype=np.float32)
            out.append(d)

    vol_ext_min, vol_ext_max = _parse_vol_params_from_extent(
        geometry,
        volume.shape,
        volume_extent_min,
        volume_extent_max)

    executor = _coneprojection(
        kernel,
        volume=volume.astype(np.float32),
        volume_extent_min=vol_ext_min,
        volume_extent_max=vol_ext_max,
        chunk_size=chunk_size,
        projections_cpu=out,
        geometries=geometry,
        **kwargs)

    for _ in tqdm(executor):
        pass

    return out


def fdk(
    projections: Any,
    geometry: Any,
    volume_shape: Sequence,
    volume_extent_min: Sequence = None,
    volume_extent_max: Sequence = None,
    chunk_size: int = 100,
    filter: Any = 'ramlak',
    preproc_fn: Callable = None,
    **kwargs):
    """Feldkamp-Davis-Kress algorithm"""

    # note: calling bp with filter
    return bp(projections,
              geometry,
              volume_shape,
              volume_extent_min,
              volume_extent_max,
              chunk_size,
              filter=filter,
              preproc_fn=preproc_fn,
              **kwargs)


def bp(
    projections: Any,
    geometry: Any,
    volume_shape: Sequence,
    volume_extent_min: Sequence = None,
    volume_extent_max: Sequence = None,
    chunk_size: int = 100,
    filter: Any = None,
    preproc_fn: Callable = None,
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

    vol_shp, vol_ext_min, vol_ext_max = _parse_vol_params_from_shape(
        geometry, volume_shape, volume_extent_min, volume_extent_max)

    executor = _conebackprojection(
        kernels.ConeBackprojection(),
        projections=projections,
        geometries=geometry,
        volume_shape=vol_shp,
        volume_extent_min=vol_ext_min,
        volume_extent_max=vol_ext_max,
        chunk_size=chunk_size,
        filter=filter,
        preproc_fn=preproc_fn,
        **kwargs)

    for volume_gpu in tqdm(executor):
        pass

    return volume_gpu.get()


def sirt_experimental(
    projections: np.ndarray,
    geometry: Any,
    volume_shape: Sequence,
    volume_extent_min: Sequence = None,
    volume_extent_max: Sequence = None,
    preproc_fn: Callable = None,
    iters: int = 100,
    dtype = cp.float32,
    verbose = True,
    **kwargs):
    """Simulateneous Iterative Reconstruction Technique

    TODO(Adriaan): There is a bug with using aspitched() and `pitch2d`
        memory allocation, which might be faster than Z-array texture. The
        bug is reproduced with a non-pitch detector dimension and the
        `aspitched` lines below uncommented. Use plotting to see weird
        backprojection residual that is probably the cause of the
        some miscomputation on wrong dimensions. For now I will use
        CUDAarray which takes a bit more memory which is also ASTRA's
        way, and just takes a bit memory.

    :param projections:
    :param geometry:
    :param volume_shape:
    :param volume_extent_min:
    :param volume_extent_max:
    :param preproc_fn:
    :param iters:
    :param dtype:
    :param kwargs:
    :return:
    """
    if len(projections) != len(geometry):
        raise ValueError

    vol_shp, vol_ext_min, vol_ext_max = _parse_vol_params_from_shape(
        geometry, volume_shape, volume_extent_min, volume_extent_max)

    # prevent copying if already in GPU memory, otherwise copy to GPU
    y = [cp.array(p, copy=False) for p in projections]
    # y_tmp = [aspitched(cp.ones_like(p)) for p in y]
    y_tmp = [cp.ones_like(p) for p in y]  # intermediate variable
    x = cp.zeros(vol_shp, dtype=dtype)  # output volume
    x_tmp = cp.ones_like(x)

    if preproc_fn is not None:
        preproc_fn(y)

    fp = kernels.ConeProjection()
    bp = kernels.ConeBackprojection()

    def A(x, out=None):
        if out is None:
            # out = [aspitched(cp.zeros_like(p)) for p in y]
            out = [cp.zeros_like(p) for p in y]
        else:
            for p in out:
                p.fill(0.)

        x_txt = _to_texture(x)
        fp(x_txt, vol_ext_min, vol_ext_max, geometry, out)
        return out

    def A_T(y, out=None):
        if out is None:
            out = cp.zeros_like(x)
        else:
            out.fill(0.)

        y_txt = [_to_texture(p) for p in y]
        bp(y_txt, geometry, out, vol_ext_min, vol_ext_max)
        return out

    # compute scaling matrix C
    C = A_T(y_tmp)
    cp.divide(1., C, out=C)  # TODO(Adriaan): there is no `where=` in CuPy
    C[C == cp.infty] = 0.

    # compute scaling operator R
    R = A(x_tmp)
    for p in R:
        cp.divide(1., p, out=p)  # TODO(Adriaan): there is no `where=` in CuPy
        p[p == cp.infty] = 0.

    # import matplotlib.pyplot as plt
    # plt.figure()
    for _ in tqdm(range(iters), disable=not verbose, desc="SIRT"):
        A(x, out=y_tmp)  # forward project `x` into `y_tmp`
        # compute residual in `y_tmp`, apply R
        for p_tmp, p, r in zip(y_tmp, y, R):
            p_tmp -= p  # residual
            p_tmp *= r

        # plt.cla()
        # plt.imshow(y_tmp[0].get(), vmax=0.1)
        # plt.pause(.15)

        # backproject residual into `x_tmp`, apply C
        A_T(y_tmp, x_tmp)
        x_tmp *= C
        x -= x_tmp  # update `x`

    return x.get()
