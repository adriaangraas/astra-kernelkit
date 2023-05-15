import collections
import copy
import warnings
from typing import Any, Callable, Sequence, Sized

import cupy as cp
import numpy as np
from tqdm import tqdm

from astrapy import kernels, process
from astrapy.geom import GeometrySequence, shift_
from astrapy.kernel import Kernel, _to_texture
from astrapy.kernels import ConeBackprojection, ConeProjection


def suggest_volume_extent(geometry, object_position: Sequence = (0., 0., 0.)):
    # TODO(Adriaan): now using only one geometry.
    #    perhaps use polygon clipping on multiple geoms
    #   to find the intersection of volume areas.
    cg = copy.deepcopy(geometry)  # TODO
    shift_(cg, -np.array(object_position))

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


def _coneprojection(
    kernel: ConeProjection,
    volume,
    volume_extent_min,
    volume_extent_max,
    geometries: Sequence,
    chunk_size: int = None,
    out: list = None,
    out_shapes: list = None,
    dtype=cp.float32,
    verbose=True,
    **kwargs):
    """
    Allocates GPU memory for only `chunk_size` projection images, then
    repeats the kernel call into the same GPU memory.

    :param kernel:
    :param chunk_size:
    :param geometries:
    :param kwargs:
    """
    volume_texture = _to_texture(volume)

    if out is None:
        assert chunk_size > 0
        with cp.cuda.stream.Stream():
            for start_proj in tqdm(range(0, len(geometries), chunk_size),
                                   desc="Forward projecting",
                                   disable=not verbose):
                next_proj = min(start_proj + chunk_size, len(geometries))
                sub_geoms = geometries[start_proj:next_proj]

                if out is not None and out_shapes is None:
                    projs_gpu = out[start_proj:next_proj]
                    [p.fill(0.) for p in projs_gpu]
                elif out_shapes is not None and out is None:
                    projs_gpu = [cp.zeros(out_shapes[i], dtype=dtype)
                                 for i in range(start_proj, next_proj)]
                else:
                    raise ValueError("Provide either `out` or `out_shapes`")

                kernel(volume_texture,
                       volume_extent_min,
                       volume_extent_max,
                       sub_geoms,
                       projs_gpu)

                yield projs_gpu
    else:
        assert chunk_size is None, "No need for chunk size!"
        [p.fill(0.) for p in out]
        kernel(volume_texture,
               volume_extent_min,
               volume_extent_max,
               geometries,
               out)

    return out


def _conebackprojection(
    kernel: ConeBackprojection,
    projections: Sequence,
    geometries: Sequence,
    volume_extent_min: Sequence,
    volume_extent_max: Sequence,
    out,
    chunk_size: int = None,
    dtype=cp.float32,
    filter: Any = None,
    preproc_fn: Callable = None,
    verbose=True,
    **kwargs):
    """
    If one of the `projections` is on CPU, use `chunk_size` to upload, process,
    compute, and download projections in batches. Alternatively, if `projections`
    lives on the GPU, compute all as a whole.
    """

    def _preproc_to_texture(projs, geoms):
        if preproc_fn is not None:
            preproc_fn(projs)
        if filter is not None:
            process.preweight(projs, geoms)
            process.filter(projs, filter=filter)
        return _to_texture(projs)

    def _compute(projs_txt, geometries):
        params = kernel.geoms2params(
            geometries,
            out.shape,
            volume_extent_min,
            volume_extent_max,
            fdk_weighting=False)
        kernel(
            projs_txt,
            params,
            out,
            volume_extent_min,
            volume_extent_max)

    # run chunk-based algorithm if one or more projections are on CPU
    if np.any([isinstance(p, np.ndarray) for p in projections]):
        assert chunk_size > 0
        with cp.cuda.stream.Stream() as stream:
            for start in tqdm(range(0, len(geometries), chunk_size),
                              desc="Backprojecting",
                              disable=not verbose):
                end = min(start + chunk_size, len(geometries))
                sub_geoms = geometries[start:end]
                sub_projs = projections[start:end]
                sub_projs_gpu = cp.asarray(sub_projs)
                projs_txt = _preproc_to_texture(sub_projs_gpu, sub_geoms)
                _compute(projs_txt, sub_geoms)
                yield out
    else:
        assert chunk_size is None, ("All `projections` are on the GPU, "
                                    "no `chunk_size` needed.")
        projections = cp.asarray(projections)
        projs_txt = _preproc_to_texture(projections, geometries)
        _compute(projs_txt, geometries)

    # TODO(Adriaan): priority, make sure this does not invoke a copy
    out[...] = cp.reshape(out, tuple(reversed(out.shape))).T
    return out


def fp(
    volume: Any,
    geometry: Any,
    volume_extent_min: Sequence = None,
    volume_extent_max: Sequence = None,
    volume_voxel_size: Sequence = None,
    chunk_size: int = 100,
    out: Sized = None,
    kernel: Kernel = None,
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
    if out is None:
        out = []
        for g in geometry:
            d = np.zeros((g.detector.rows, g.detector.cols), dtype=np.float32)
            out.append(d)

    _, vol_ext_min, vol_ext_max, _ = vol_params(
        volume.shape,
        volume_extent_min,
        volume_extent_max,
        volume_voxel_size,
        geometry,
    )

    if kernel is None:
        kernel = ConeProjection()

    executor = _coneprojection(
        kernel,
        volume=volume.astype(np.float32),
        volume_extent_min=vol_ext_min,
        volume_extent_max=vol_ext_max,
        chunk_size=chunk_size,
        geometries=geometry,
        out_shapes=[p.shape for p in out],
        **kwargs)

    i = 0
    for batch in tqdm(executor):
        for p in batch:
            out[i][...] = p.get()
            i += 1

    return out


def fdk(
    projections: Any,
    geometry: Any,
    volume_shape: Sequence,
    volume_extent_min: Sequence = None,
    volume_extent_max: Sequence = None,
    volume_voxel_size: Sequence = None,
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
              volume_voxel_size,
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
    volume_voxel_size: Sequence = None,
    chunk_size: int = 100,
    filter: Any = None,
    preproc_fn: Callable = None,
    return_gpu: bool = False,
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
    if len(projections) != len(geometry):
        raise ValueError

    if chunk_size is None:
        # TODO: allow None and auto infer good chunk size
        raise NotImplementedError()

    vol_shp, vol_ext_min, vol_ext_max, _ = vol_params(
        volume_shape, volume_extent_min, volume_extent_max,
        volume_voxel_size, geometry, verbose=verbose)

    if out is None:
        volume_gpu = cp.empty(vol_shp, dtype=cp.float32)
    else:
        volume_gpu = out

    if kernel is None:
        kernel = kernels.ConeBackprojection()

    executor = _conebackprojection(
        kernel,
        projections=projections,
        geometries=geometry,
        volume_extent_min=vol_ext_min,
        volume_extent_max=vol_ext_max,
        out=volume_gpu,
        chunk_size=chunk_size,
        filter=filter,
        preproc_fn=preproc_fn,
        verbose=verbose,
        **kwargs)

    for volume_gpu in tqdm(executor, disable=not verbose):
        pass

    if return_gpu:
        return volume_gpu

    return volume_gpu.get()


class _A:
    def __init__(self,
                 fpkern: ConeProjection,
                 vol_ext_min,
                 vol_ext_max,
                 geometry,
                 out_shapes,
                 chunk_size=None,
                 xp_out=cp,
                 dtype=cp.float32,
                 verbose=False):
        self.fpk = fpkern
        self.vol_ext_min = vol_ext_min
        self.vol_ext_max = vol_ext_max
        self.geometry = GeometrySequence.fromList(geometry)
        self.out_shapes = out_shapes
        self.xp = xp_out
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.verbose = verbose

    def __call__(self, x, out=None):
        """
        Out not given:
            xp = np: make xp array, call with None, draw output in new array
            xp = cp: make xp array, call with xp array, draw no output
        Out given:
            out np: call with None, draw output in given array
            out cp: call with given array, return
        :param x:
        :param out:
        :return:
        """
        if out is None:
            # out = [aspitched(cp.zeros_like(p)) for p in y]
            out = [self.xp.zeros(shp, dtype=self.dtype) for shp in
                   self.out_shapes]
        # else:
        #     for p in out:
        #         p.fill(0.)

        exc = _coneprojection(
            self.fpk,
            x,
            self.vol_ext_min,
            self.vol_ext_max,
            chunk_size=self.chunk_size,
            geometries=self.geometry,
            out=out if self.xp == cp else None,
            out_shapes=self.out_shapes if self.xp == np else None,
            dtype=self.dtype,
            verbose=self.verbose)

        i = 0
        for out_projs in exc:
            for q in out_projs:
                if cp.get_array_module(out[i]) == np:
                    out[i][...] = q.get()
                i += 1

        # self.fpk.__call__(
        #     _to_texture(x),
        #     self.vol_ext_min,
        #     self.vol_ext_max,
        #     self.geometry,
        #     out,
        # )

        return out


class _A_T:
    def __init__(self, bpkern: ConeBackprojection,
                 vol_ext_min, vol_ext_max, geometry,
                 out_shape, chunk_size=None, xp_out=cp, dtype=cp.float32,
                 verbose=False):
        self.bpk = bpkern
        self.vol_ext_min = vol_ext_min
        self.vol_ext_max = vol_ext_max
        self.geometry = GeometrySequence.fromList(geometry)
        self.out_shape = out_shape
        self.xp = xp_out
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.verbose = verbose

    def __call__(self, y, out=None):
        """
        Out given, GPU     : produce in out
        Out given, CPU     : cp create array, use .get()
        Out not given, GPU : cp create array, return that
        Out not given, CPU : cp create array, np create array, .get()
        """
        if out is not None:
            if cp.get_array_module(out) == np:
                gpu_out = cp.zeros(out.shape, self.dtype)
            else:
                gpu_out = out
        else:
            gpu_out = cp.zeros(self.out_shape, self.dtype)

        # y_txt = [_to_texture(p) for p in y]
        # y_txt = []
        # for p in y:  # synchronous convertion to texture and deleteion
        #     y_txt += [_to_texture(p)]
        #     del p
        #
        # del y  # just in case

        exc = _conebackprojection(
            self.bpk,
            y,
            self.geometry,
            self.vol_ext_min,
            self.vol_ext_max,
            gpu_out,
            chunk_size=self.chunk_size,
            dtype=self.dtype,
            verbose=self.verbose)

        for _ in exc:
            pass

        if self.xp == cp:
            out = gpu_out
        else:
            out = gpu_out.get()

        return out


def sirt_experimental(
    projections: np.ndarray,
    geometry: Any,
    volume_shape: Sequence,
    volume_extent_min: Sequence = None,
    volume_extent_max: Sequence = None,
    volume_voxel_size: Sequence = None,
    preproc_fn: Callable = None,
    iters: int = 100,
    dtype=cp.float32,
    verbose: bool = True,
    mask: Any = None,
    proj_mask: Any = True,
    x_0: Any = None,
    min_constraint: float = None,
    max_constraint: float = None,
    return_gpu: bool = False,
    chunk_size=300,
    algo='gpu',
    callback: callable=None):
    """Simulateneous Iterative Reconstruction Technique

    TODO(Adriaan): There is a bug with using aspitched() and `pitch2d`
        memory allocation, which might be faster than Z-array texture. The
        bug is reproduced with a non-pitch detector dimension and the
        `aspitched` lines below uncommented. Use plotting to see weird
        backprojection residual that is probably the cause of the
        some miscomputation on wrong dimensions. For now I will use
        CUDAarray which takes a bit more memory which is also ASTRA's
        way.

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
        op_kwargs = {'verbose': verbose, 'chunk_size': chunk_size}
    elif algo == 'gpu':
        xp_proj = cp
        op_kwargs = {'verbose': verbose}
    else:
        raise ValueError

    xp_vol = cp

    # prevent copying if already in GPU memory, otherwise copy to GPU
    y = [xp_proj.array(p, copy=False) for p in projections]

    if x_0 is not None:
        x = xp_vol.asarray(x_0, dtype)
    else:
        x = xp_vol.zeros(vol_shp, dtype)  # output volume

    x_tmp = xp_vol.ones_like(x)

    if preproc_fn is not None:
        preproc_fn(y)

    prj_shps = [p.shape for p in projections]
    A = _A(kernels.ConeProjection(), vol_ext_min, vol_ext_max, geometry,
           prj_shps, xp_out=xp_proj, **op_kwargs)
    A_T = _A_T(kernels.ConeBackprojection(), vol_ext_min, vol_ext_max, geometry,
               vol_shp, xp_out=xp_vol, **op_kwargs)

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

    bar = tqdm(range(iters), disable=not verbose, desc="SIRT (starting...)")
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

        if i % 10 == 0:  # this operation is expensive
            res_norm = str(sum([xp_proj.linalg.norm(p) for p in y_tmp]))
        bar.set_description(f"SIRT (update in {10 - i % 10}): {res_norm})")

        # backproject residual into `x_tmp`, apply C
        x_tmp = A_T(y_tmp)
        x_tmp *= C
        x -= x_tmp  # update `x`
        if mask is not None:
            x *= mask
        if min_constraint is not None or max_constraint is not None:
            xp_vol.clip(x, a_min=min_constraint, a_max=max_constraint, out=x)

        if callback is not None:
            callback(i, x, y_tmp)

        cp.get_default_memory_pool().free_all_blocks()

    if xp_vol == cp and not return_gpu:
        return x.get()

    return x
