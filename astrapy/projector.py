from typing import Any, Callable
import cupy as cp
import numpy as np

import astrapy as ap


class ConeProjector:
    def __init__(self,
                 kernel: ap.ConeProjection = None,
                 dtype=cp.float32,
                 verbose=True):
        self._K = ap.ConeProjection() if kernel is None else kernel
        self._dtype = dtype
        self._verbose = verbose
        self._volume_texture = None
        self._volume = None
        self._projections = None

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
        self._geometry_seq = ap.GeometrySequence.fromList(self._geometry)
        # TODO precalculate params here

    def _out_shape_from_geometry(self):
        nr_angles = len(self._geometry)
        det_shape = (self._geometry[0].detector.rows,
                     self._geometry[0].detector.cols)
        return (nr_angles, *det_shape)

    def __call__(self, ext_min, ext_max, out=None, additive=False):
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
                ext_min,
                ext_max,
                self._geometry_seq,
                out)
        return out


class ConeBackprojector:
    def __init__(self,
                 kernel: ap.ConeBackprojection = None,
                 filter: Any = None,
                 preproc_fn: Callable = None,
                 texture_type: str = 'array',
                 dtype=cp.float32,
                 verbose=True):
        """
        If one of the `projections` is on CPU, use `chunk_size` to upload, process,
        compute, and download projections in batches. Alternatively, if `projections`
        lives on the GPU, compute all as a whole.
        """
        self._K = ap.ConeBackprojection() if kernel is None else kernel
        self._dtype = dtype
        self._verbose = verbose
        self._texture_type = texture_type.lower()
        if self._texture_type not in ('array', 'pitch2d'):
            raise ValueError("`texture_type` not supported.")
        self._filter = filter
        self._preproc_fn = preproc_fn
        self._geometry_seq = None
        self._geometry_params = None
        self._geometry_params_shp = None
        self._geometry_params_ext_min = None
        self._geometry_params_ext_max = None
        self._projections_txt = None
        self._projections = None

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        self._geometry = value
        self._geometry_seq = ap.GeometrySequence.fromList(self._geometry)

    def _params(self, vol_shp, ext_min, ext_max):
        if (self._geometry_params is None or
            (vol_shp == self._geometry_params_shp and
             ext_min == self._geometry_params_ext_min and
             ext_max == self._geometry_params_ext_max)):
            if self._geometry_seq is None:
                raise RuntimeError("Make sure to set the `geometry` property"
                                   " of this projector before calling it.")
            self._geometry_params = self._K.geoms2params(
                self._geometry_seq,
                vol_shp,
                ext_min,
                ext_max)
        return self._geometry_params

    @property
    def projections(self):
        return self._projections

    @projections.setter
    def projections(self, value):
        self._projections = value
        if self._texture_type == 'array':
            self._projections_txt = None  # invalidate textures if CUDA array

    def _projection_textures(self):
        if self._projections_txt is None:
            if self._projections is None:
                raise RuntimeError(
                    "Make sure to set the `projections` property"
                    " of this projector before calling it.")
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

    def __call__(self, volume_extent_min, volume_extent_max,
                 volume_shape=None, additive=False,
                 out=None):
        if out is None:
            assert additive is False, "Setting `additive` has no effect."
            if volume_shape is None:
                raise RuntimeError("Provide either `volume_shape` or `out`.")
            out = cp.zeros(volume_shape, dtype=self._dtype)
        else:
            assert volume_shape is None, \
                "Setting `vol_shp` has no effect when `out` is provided."
            if not additive:
                out.fill(0.)
        volume_shape = out.shape

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
            self._projection_textures(),
            self._params(volume_shape, volume_extent_min, volume_extent_max),
            out,
            volume_extent_min,
            volume_extent_max)

        # TODO(Adriaan): priority, make sure this does not invoke a copy
        out[...] = cp.reshape(out, tuple(reversed(out.shape))).T
        return out


def astra_compat_geoms2vectors(geoms):
    def geom2astra(g):
        c = lambda x: (x[1], x[0], x[2])
        d = lambda x: np.array((x[1], x[0], x[2])) * g.detector.pixel_width
        e = lambda x: np.array((x[1], x[0], x[2])) * g.detector.pixel_height
        return [*c(g.source_position), *c(g.detector_position),
                *d(g.u), *e(g.v)]
    vectors = np.zeros((len(geoms), 12))
    for i, g in enumerate(geoms):
        vectors[i] = geom2astra(g)
    return vectors


class AstraCompatConeProjector:
    def __init__(self):
        self._geometry = None
        self._geometry_seq = None
        self._volume_texture = None
        self._volume = None
        self._projector_id = None
        self._projector_id_proj_shp = None
        self._projector_id_ext_min = None
        self._projector_id_ext_max = None

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, value):
        # make sure CuPy doesn't clean up while ASTRA is busy
        self._volume = value

        import astra
        vol = cp.ascontiguousarray(cp.transpose(self._volume, (2, 0, 1)))
        z, y, x = vol.shape
        self._vol_link = astra.pythonutils.GPULink(
            vol.data.ptr, x, y, z, x * 4)
        # TODO: delete old projector and volume
        self._projector_id = None

    def projector3d(self, ext_min, ext_max):
        if (self._projector_id is None or (
            np.all(ext_min == self._projector_id_ext_min) and
            np.all(ext_max == self._projector_id_ext_max))):
            import astra
            if self.volume is None:
                raise ValueError("Set `volume` first.")
            vol_geom = astra.create_vol_geom(
                *self._volume.shape,
                ext_min[1], ext_max[1],
                ext_min[0], ext_max[0],
                ext_min[2], ext_max[2])

            if self.geometry is None:
                raise ValueError("Set `geometry` first.")
            vectors = astra_compat_geoms2vectors(self.geometry)
            proj_shp = (self._geometry_seq.detector.rows[0],
                        self._geometry_seq.detector.cols[0])

            proj_geom = astra.create_proj_geom('cone_vec', *proj_shp, vectors)
            proj_cfg = {'type': 'cuda3d',
                        'VolumeGeometry': vol_geom,
                        'ProjectionGeometry': proj_geom,
                        'options': {}}
            self._projector_id = astra.projector3d.create(proj_cfg)
        return self._projector_id

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        self._geometry = value
        self._geometry_seq = ap.GeometrySequence.fromList(self._geometry)

    def _out_shape_from_geometry(self):
        if self._geometry is None:
            raise ValueError("Set `geometry` first.")

        nr_angles = len(self._geometry)
        det_shape = (self._geometry[0].detector.rows,
                     self._geometry[0].detector.cols)
        return (nr_angles, *det_shape)

    def __call__(self, volume_extent_min, volume_extent_max,
                 out=None, additive=False):
        import astra
        import astra.experimental

        if out is None:
            out = cp.zeros(
                self._out_shape_from_geometry(),
                dtype=cp.float32)

        projections = cp.ascontiguousarray(
            out.transpose(1, 0, 2))
        z, y, x = projections.shape
        proj_link = astra.pythonutils.GPULink(
            projections.data.ptr,
            x, y, z, x * 4)
        astra.experimental.direct_FPBP3D(
            self.projector3d(volume_extent_min, volume_extent_max),
            self._vol_link,
            proj_link,
            int(not additive),
            "FP")
        out[...] = projections.transpose(1, 0, 2)
        return out


class AstraCompatConeBackprojector:
    def __init__(self):
        """
        If one of the `projections` is on CPU, use `chunk_size` to upload, process,
        compute, and download projections in batches. Alternatively, if `projections`
        lives on the GPU, compute all as a whole.
        """
        self._proj_link = None
        self._projections = None
        self._proj_geom = None
        self._projector_id = None

    @property
    def projections(self):
        return self._projections

    @projections.setter
    def projections(self, value: cp.ndarray):
        import astra
        # needs to remain an attribute otherwise cleaned up
        value = cp.asarray(value, cp.float32)
        self._proj = cp.ascontiguousarray(value.transpose(1, 0, 2), cp.float32)
        z, y, x = self._proj.shape
        self._proj_link = astra.pythonutils.GPULink(
            self._proj.data.ptr, x, y, z, x * 4)
        # TODO delete ASTRA prjoector and projecitons
        self._projector_id = None

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        import astra
        self._geometry = value
        vectors = astra_compat_geoms2vectors(value)
        det_shape = (self._geometry[0].detector.rows,
                     self._geometry[0].detector.cols)
        self._proj_geom = astra.create_proj_geom(
            'cone_vec', *det_shape, vectors)
        self._projector_id = None  # invalidate old projector
        # TODO delete

    def projector3d(self, vol_shp, ext_min, ext_max):
        import astra
        if self._projector_id is None:
            if self._proj_geom is None:
                raise RuntimeError("Geometry must be given before creating "
                                   "a projector. Use the `geometry` setter "
                                   "of this class.")
            self._vol_geom = astra.create_vol_geom(
                *vol_shp,
                ext_min[1], ext_max[1],
                ext_min[0], ext_max[0],
                ext_min[2], ext_max[2])
            proj_cfg = {'type': 'cuda3d',
                        'VolumeGeometry': self._vol_geom,
                        'ProjectionGeometry': self._proj_geom,
                        'options': {}}
            self._projector_id = astra.projector3d.create(proj_cfg)
        return self._projector_id

    def __call__(self, volume_extent_min, volume_extent_max,
                 volume_shape=None, additive=False, out=None):
        if out is None:
            assert additive is False, "Setting `additive` has no effect."
            if volume_shape is None:
                raise RuntimeError("Provide either `volume_shape` or `out`.")
            out = cp.zeros(volume_shape, dtype=cp.float32)
        else:
            assert volume_shape is None, \
                "Setting `vol_shp` has no effect when `out` is provided."
            if not additive:
                out.fill(0.)
        volume_shape = out.shape

        import astra.experimental
        astra_vol = cp.ascontiguousarray(cp.transpose(out, (2, 0, 1)))
        z, y, x = astra_vol.shape
        vol_link = astra.pythonutils.GPULink(astra_vol.data.ptr, x, y, z,
                                             x * 4)
        astra.experimental.direct_FPBP3D(
            self.projector3d(volume_shape, volume_extent_min,
                             volume_extent_max),
            vol_link,
            self._proj_link,
            int(not additive),
            "BP")
        out[...] = astra_vol.transpose(1, 2, 0)
        return out
