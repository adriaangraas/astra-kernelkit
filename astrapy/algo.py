from typing import Any, Callable, Sequence, Sized

import cupy as cp
import numpy as np
from tqdm import tqdm

from astrapy import kernels, process
from astrapy.data import aspitched, voxel_size
from astrapy.kernel import Kernel, _copy_to_texture


def fp(
    volume: Any,
    geometry: Any,
    projections_cpu: Sized = None,
    volume_extent_min: Sequence = (-1, -1, -1),
    volume_extent_max: Sequence = (1, 1, 1),
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


class ProjectionBufferBp:
    """Reconstructions from a GPU buffer"""

    def __init__(
        self,
        kernel: Kernel,
        projection_ids: Sequence,
        geometries: list,
        fload: Callable,
        filter: bool = True,
        fpreproc: Callable = None,
        texture_type: str = 'pitch2D',
        minimum_load_size: int = 250,
        verbose: bool = False):
        self.verbose = verbose
        self._texture_type = texture_type
        self._projection_ids = projection_ids
        self._kernel = kernel
        self._geoms = geometries
        self._filter = filter
        self._fpreproc = fpreproc
        self._fload = fload
        self._textures = {}
        self._min_load = minimum_load_size

    def _load_proj(self, ids):
        ids_not_loaded = list(
            filter(lambda i: i not in self._textures.keys(), ids))
        if len(ids_not_loaded) == 0:
            return [self._textures[i] for i in ids]
        else:
            if len(ids_not_loaded) < self._min_load:
                # find ourselves `_min_load` unloaded projection ids
                for i in self._projection_ids:
                    if i not in self._textures.keys() and i not in ids_not_loaded:
                        ids_not_loaded.append(i)
                    if len(ids_not_loaded) == self._min_load:
                        break

        projections_cpu = self._fload(ids_not_loaded)
        projs_gpu = cp.asarray(projections_cpu, dtype=cp.float32)
        if self._fpreproc is not None:
            self._fpreproc(projs_gpu)

        # this is why geometries have to be given at __init__
        geoms = [self._geoms[id] for id in ids_not_loaded]
        process.preweight(projs_gpu, geoms)
        if self._filter:
            process.filter(projs_gpu, verbose=self.verbose)

        for i, p in zip(ids_not_loaded, projs_gpu):
            p = aspitched(p) if self._texture_type == 'pitch2D' else p
            self._textures[i] = \
                _copy_to_texture(aspitched(cp.asarray(p)),
                                 type=self._texture_type)

        return [self._textures[i] for i in ids]

    def __call__(self,
                 numbers: Sequence,
                 volume_shape: Sequence,
                 volume_extent_min: Sequence,
                 volume_extent_max: Sequence,
                 volume_rotation: Sequence,
                 dtype=cp.float32):
        if not all(i in range(len(self._projection_ids)) for i in numbers):
            raise ValueError(
                "`numbers` have to be indices of the projections.")

        with cp.cuda.Stream(non_blocking=True):
            textures = self._load_proj(numbers)

        with cp.cuda.Stream(null=True) as s:
            # TODO: recycle?
            volume = cp.zeros(tuple(volume_shape), dtype=dtype)
            self._kernel(
                textures,
                [self._geoms[id] for id in numbers],
                volume,
                volume_extent_min,
                volume_extent_max,
                volume_rotation)

        with cp.cuda.Stream(null=True, non_blocking=False) as s:
            vol = volume.get()
            s.synchronize()

        return vol

    def slice(self,
              numbers: Sequence,
              slice_shape: Sequence,
              slice_extent_min: Sequence,
              slice_extent_max: Sequence,
              slice_rotation: Sequence = (0., 0., 0.),
              slice_position: Sequence = (0., 0., 0.),
              **kwargs):
        vol_shape = (*slice_shape, 1)
        # this preserves isotropic voxels, and otherwise chooses a thickness
        # equivalent to the smallest voxel dim
        vox_size = voxel_size(slice_shape, slice_extent_min, slice_extent_max)
        s = np.min(vox_size)
        vol_ext_min = (*slice_extent_min, -s / 2)
        vol_ext_max = (*slice_extent_max, s / 2)
        vol_ext_min = np.array(vol_ext_min) - slice_position
        vol_ext_max = np.array(vol_ext_max) - slice_position
        vol = self(numbers, vol_shape, vol_ext_min, vol_ext_max,
                   slice_rotation,
                   **kwargs)
        return np.squeeze(vol)

    def randslice(self,
                  numbers_range: range,
                  numbers_length: int,
                  slice_shape,
                  slice_extent_min,
                  slice_extent_max,
                  slice_rotation: Sequence = (0., 0., 0.),
                  slice_position_min: Sequence = (0., 0., 0.),
                  slice_position_max: Sequence = (0., 0., 0.),
                  numbers_step: int = 1):
        assert numbers_range.step == 1, "Not implemented."
        slice_num_start = np.random.randint(numbers_range.start,
                                            numbers_range.stop - numbers_length)
        slice_num_stop = slice_num_start + numbers_length
        space = np.subtract(slice_position_max, slice_position_min)
        assert np.all(space >= 0)
        pos = np.random.random_sample(3) * space + slice_position_min
        rot = np.random.random_sample(3) * slice_rotation
        return self.slice(
            range(slice_num_start, slice_num_stop, numbers_step),
            slice_shape,
            slice_extent_min,
            slice_extent_max,
            slice_rotation=rot,
            slice_position=pos)
