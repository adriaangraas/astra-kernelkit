from typing import Any, Callable, Sequence

from tqdm import tqdm
import cupy as cp
import astrapy as ap


class ConeProjector:
    def __init__(self,
                 kernel: ap.ConeProjection,
                 volume_extent_min,
                 volume_extent_max,
                 dtype=cp.float32,
                 verbose=True):
        """
        Allocates GPU memory for only `chunk_size` projection images, then
        repeats the kernel call into the same GPU memory.

        :param kernel:
        :param chunk_size:
        :param geometries:
        :param kwargs:
        """
        self._volume_texture = None
        self._K = kernel
        self._ext_min = volume_extent_min
        self._ext_max = volume_extent_max
        self._dtype = dtype
        self._verbose = verbose
        self._volume = None

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = value
        self._volume_texture = ap.copy_to_texture(value)

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        self._geometry = value
        self._geometry_list = ap.GeometrySequence.fromList(self._geometry)

    def _out_shape_from_geometry(self):
        nr_angles = len(self._geometry)
        det_shape = (self._geometry[0].detector.rows,
                     self._geometry[0].detector.cols)
        return (nr_angles, *det_shape)

    def __call__(self, out=None, additive=False):
        if out is None:
            out = cp.zeros(self._out_shape_from_geometry(),
                           dtype=self._dtype)

        #     for start_proj in tqdm(
        #         range(0, len(self.geometry), self._chunk_size),
        #         desc="Forward projecting",
        #         disable=not self._verbose):
        #         next_proj = min(start_proj + self._chunk_size,
        #                         len(self.geometry))
        #         sub_geoms = self.geometry[start_proj:next_proj]
        #
        #         if out is not None and self._out_shapes is None:
        #             projs_gpu = out[start_proj:next_proj]
        #             [p.fill(0.) for p in projs_gpu]
        #         elif self._out_shapes is not None and out is None:
        #             projs_gpu = [
        #                 cp.zeros(self._out_shapes[i], dtype=self._dtype)
        #                 for i in range(start_proj, next_proj)]
        #         else:
        #             raise ValueError(
        #                 "Provide either `out` or `out_shapes`")

        if not additive:
            [p.fill(0.) for p in out]

        self._K(self._volume_texture,
                self._ext_min,
                self._ext_max,
                self._geometry_list,
                out)

        return out


class ConeBackprojector:
    def __init__(self,
                 kernel: ap.ConeBackprojection,
                 volume_shape,
                 volume_extent_min,
                 volume_extent_max,
                 filter: Any = None,
                 preproc_fn: Callable = None,
                 texture_type: str='array',
                 dtype=cp.float32,
                 verbose=True):
        """
        If one of the `projections` is on CPU, use `chunk_size` to upload, process,
        compute, and download projections in batches. Alternatively, if `projections`
        lives on the GPU, compute all as a whole.
        """
        self._K = kernel
        self._vol_shp = volume_shape
        self._ext_min = volume_extent_min
        self._ext_max = volume_extent_max
        self._dtype = dtype
        self._verbose = verbose
        self._texture_type = texture_type.lower()
        if self._texture_type not in ('array', 'pitch2d'):
            raise ValueError("`texture_type` not supported.")
        self._filter = filter
        self._preproc_fn = preproc_fn
        self._projections_txt = None
        self._projections = None

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        self._geometry = value
        self._geometry_list = ap.GeometrySequence.fromList(self._geometry)
        self._params = self._K.geoms2params(
            self._geometry_list,
            self._vol_shp,
            self._ext_min,
            self._ext_max)

    @property
    def projections(self):
        return self._projections

    @projections.setter
    def projections(self, value):
        self._projections = value
        if self._texture_type == 'array':
            self._projections_txt = None  # invalidate textures if CUDA array

    @property
    def _projection_textures(self):
        if self._projections_txt is None:
            if self._texture_type.lower() == 'pitch2d':
                projs = [ap.aspitched(p, cp) for p in self._projections]
            else:
                projs = cp.asarray(self.projections)

            if self._preproc_fn is not None:
                self._preproc_fn(projs)
            if self._filter is not None:
                ap.preweight(projs, self._geometry)
                ap.filter(projs, filter=self._filter)

            if self._texture_type.lower() == 'pitch2d':
                txt = [ap.copy_to_texture(p, self._texture_type) for p in
                        projs]
            else:
                txt = ap.copy_to_texture(projs, self._texture_type)
            self._projections_txt = txt

        return self._projections_txt

    def __call__(self, out=None, additive=False):
        if out is None:
            assert additive is False, "Setting `additive` has no effect."
            out = cp.zeros(self._vol_shp, dtype=self._dtype)

        # run chunk-based algorithm if one or more projections are on CPU
        # with cp.cuda.stream.Stream():
        #     if self._chunk_size is None:
        #         blocks = range(0, 1)
        #         chunk_size = len(self._geometry)
        #     else:
        #         blocks = range(0, len(self._geometry), self._chunk_size)
        #
        #     for start in tqdm(blocks, desc="Backprojecting",
        #                       disable=not self._verbose):
        #         end = min(start + chunk_size, len(self.geometry))
        #         sub_geoms = self._geometry_list[start:end]
        #         sub_projs = projections[start:end]
        self._K(
            self._projection_textures,
            self._params,
            out,
            self._ext_min,
            self._ext_max)

        # TODO(Adriaan): priority, make sure this does not invoke a copy
        out[...] = cp.reshape(out, tuple(reversed(out.shape))).T
        return out