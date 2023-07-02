from typing import Any, Callable, List, Optional, Tuple
import cupy as cp
import numpy as np
from abc import ABC

from astrapy.process import filter, preweight
from astrapy.data import aspitched
from astrapy.geom.proj import GeometrySequence, ProjectionGeometry
from astrapy.geom.vol import VolumeGeometry
from astrapy.kernel import copy_to_texture
from astrapy.kernels.cone_bp import ConeBackprojection
from astrapy.kernels.cone_fp import ConeProjection

import astra
import astra.experimental


class Projector(ABC):
    def __init__(self):
        self._vol_geom: Optional[VolumeGeometry] = None
        self._proj_geom: Optional[List[ProjectionGeometry]] = None

    @property
    def projection_geometry(self) -> Optional[List[ProjectionGeometry]]:
        return self._proj_geom

    @projection_geometry.setter
    def projection_geometry(self, value: List[ProjectionGeometry]):
        del self.projection_geometry
        self._proj_geom = value
        self._proj_geom_seq = GeometrySequence.fromList(self._proj_geom)

    @projection_geometry.deleter
    def projection_geometry(self):
        self._proj_geom = None
        self._proj_geom_seq = None

    @property
    def volume_geometry(self) -> Optional[VolumeGeometry]:
        return self._vol_geom

    @volume_geometry.setter
    def volume_geometry(self, value: VolumeGeometry):
        self._vol_geom = value
        # TODO invalidate params here


class ConeProjector(Projector):
    def __init__(self, kernel: ConeProjection = None, dtype=cp.float32,
                 verbose=True):
        super().__init__()
        self._K = ConeProjection() if kernel is None else kernel
        self._dtype = dtype
        self._verbose = verbose
        self._vol = None
        self._projs = None
        self._vol_txt = None
        self._graph: Optional[cp.cuda.Graph] = None

    @property
    def volume(self) -> Optional[cp.ndarray]:
        return self._vol

    @volume.setter
    def volume(self, value):
        self._vol = value
        self._vol_txt = copy_to_texture(value)

    @property
    def projections(self) -> Optional[cp.ndarray]:
        return self._projs

    @projections.setter
    def projections(self, value: cp.ndarray):
        self._projs = value

    @volume.deleter
    def volume(self):
        self._vol = None
        self._vol_txt = None

    @property
    def projection_shape(self) -> Tuple:
        nr_angles = len(self._proj_geom)
        det_shape = (self._proj_geom[0].detector.rows,
                     self._proj_geom[0].detector.cols)
        return nr_angles, *det_shape

    def __call__(self, additive=False, stream=None):
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
            [p.fill(0.) for p in self._projs]
        self._K(self._vol_txt,
                self._vol_geom,
                self._proj_geom_seq,
                self._projs)
        return self._projs


class ConeBackprojector(Projector):
    def __init__(self, kernel: ConeBackprojection = None, filter: Any = None,
                 preproc_fn: Callable = None, texture_type: str = 'array',
                 dtype=cp.float32, volume_axis=(0, 1, 2), verbose=True):
        """
        If one of the `projections` is on CPU, use `chunk_size` to upload, process,
        compute, and download projections in batches. Alternatively, if `projections`
        lives on the GPU, compute all as a whole.
        """
        super().__init__()
        self._K = ConeBackprojection() if kernel is None else kernel
        self._vol_axs = volume_axis
        self._dtype = dtype
        self._verbose = verbose
        self._texture_type = texture_type.lower()
        if self._texture_type not in ('array', 'pitch2d'):
            raise ValueError("`texture_type` not supported.")
        self._filter = filter
        self._preproc_fn = preproc_fn
        self._textures = None
        self._params = None
        self._projections = None
        self._graph: Optional[cp.cuda.Graph] = None

    @property
    def volume(self) -> Optional[cp.ndarray]:
        return self._vol

    @volume.setter
    def volume(self, value):
        self._vol = value

    @property
    def projections(self):
        return self._projections

    @projections.setter
    def projections(self, value):
        del self.projections
        self._projections = value

    @projections.deleter
    def projections(self):
        self._projections = None
        if self._texture_type == 'array':
            self._textures = None  # invalidate textures if CUDA array

    def _compute_params(self):
        """Compute the parameters for the backprojection kernel."""

        if self._params is None:
            if self._proj_geom_seq is None:
                raise RuntimeError("Make sure to set the `geometry` property"
                                   " of this projector before calling it.")
            self._params = self._K.geoms2params(self._proj_geom_seq,
                                                self.volume_geometry)
        return self._params

    def _compute_textures(self):
        """Compute the textures for the backprojection kernel."""

        if self._textures is None:
            if self._projections is None:
                raise RuntimeError(
                    "Make sure to set the `projections` property"
                    " of this projector before calling it.")
            if self._texture_type.lower() == 'pitch2d':
                projs = [aspitched(p, cp) for p in self._projections]
            else:
                projs = cp.asarray(self.projections)

            if self._preproc_fn is not None:
                self._preproc_fn(projs)
            if self._filter is not None:
                preweight(projs, self._proj_geom)
                filter(projs, filter=self._filter)

            if self._texture_type.lower() == 'pitch2d':
                txt = [copy_to_texture(p, self._texture_type) for p in
                       projs]
            else:
                txt = copy_to_texture(projs, self._texture_type)
            self._textures = txt

        return self._textures

    def __call__(self, additive=False):
        # if out is None:
        #     assert not additive, "Setting `additive` has no effect."
        #     out = cp.zeros(self.volume_geometry.shape, dtype=self._dtype)
        # else:
        #     assert out.shape == tuple(self.volume_geometry.shape)

        if not additive:
            self.volume.fill(0.)

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
        txts = self._compute_textures()
        params = self._compute_params()
        # TODO: reset graph when parameters change

        # if self._graph is not None:
        #     print('relaunch')
        #     self._graph.launch(stream)
        #     return out
        #
        # stream = cp.cuda.get_current_stream()
        # if stream != stream.null:
        #     stream.begin_capture()
        self._K(txts, params, self._vol, self.volume_geometry)

        if self._vol_axs == (0, 1, 2):
            # TODO(Adriaan): priority, make sure this does not invoke a copy
            self._vol[...] = cp.reshape(self._vol,
                                        tuple(reversed(self._vol.shape))).T
        elif self._vol_axs == (2, 1, 0):
            pass
        else:
            raise NotImplementedError

        # if stream.is_capturing():
        #     self._graph = stream.end_capture()
        #     self._graph.launch()

        return self._vol


class CompatProjector(Projector, ABC):
    """Base class for ASTRA projectors."""

    def __init__(self):
        """Base class for ASTRA projectors."""
        super().__init__()

        self._volume = None
        self._projections = None
        self._vol_geom = None
        self._proj_geom = None
        self._astra_vol_link = None
        self._astra_proj_link = None
        self._astra_vol_geom = None
        self._astra_proj_geom = None
        self._astra_projector_id = None

    @property
    def volume(self) -> cp.ndarray:
        """The volume to project from."""
        return self._volume

    @volume.setter
    def volume(self, value: cp.ndarray):
        """The volume to project from.

        If the volume is of the same shape as before, the ASTRA projector
        will not be reinitialized.

        Parameters
        ----------
        value : array-like
            The volume to project from."""
        if self.volume_geometry is not None:
            expected_shape = tuple(reversed(self.volume_geometry.shape))
            if expected_shape != value.shape:
                raise ValueError(
                    f"Expected volume of shape {expected_shape} (z, y, x)"
                    f" according to the volume geometry, but got "
                    f"{value.shape} instead.")

        if self._volume is not None and self._volume.shape != value.shape:
            self._astra_projector_id = None

        # make sure CuPy doesn't clean up the volume while ASTRA is busy
        assert value.dtype == cp.float32
        assert value.flags.c_contiguous
        self._volume = value
        z, y, x = self._volume.shape
        self._astra_vol_link = astra.pythonutils.GPULink(
            self._volume.data.ptr, x, y, z, x * 4)

    @volume.deleter
    def volume(self):
        """Delete the volume and clean up ASTRA projector."""
        self._volume = None
        self._astra_vol_link = None  # TODO: does this delete
        print("Deleting projector via volume")
        self._astra_projector_id = None

    @property
    def projections(self) -> cp.ndarray:
        """The projections onto which to project."""
        return self._projections

    @projections.setter
    def projections(self, value: cp.ndarray):
        """The projections onto which to project."""
        assert value.dtype == cp.float32
        assert value.flags.c_contiguous

        if self.projection_geometry is not None:
            expected_shape = (self.projection_geometry[0].detector.rows,
                              len(self.projection_geometry),
                              self.projection_geometry[0].detector.cols)
            if value.shape != expected_shape:
                raise ValueError(f"Projection shape does not match projection"
                                 f" geometry. Got {value.shape}, "
                                 f" expected {expected_shape}.")

        if self._projections is not None \
            and self._projections.shape != value.shape:
            self._astra_projector_id = None

        # maintain proj_link if it exists
        z, y, x = value.shape
        if (self._astra_proj_link is not None
            and self._projections.shape == value.shape):
            self._projections[...] = value  # update in-place
        else:
            self._projections = value
            self._astra_proj_link = astra.pythonutils.GPULink(
                self._projections.data.ptr, x, y, z, x * 4)
            # needs to remain a class attribute, otherwise cleaned up

    @projections.deleter
    def projections(self):
        """Delete the projections and clean up ASTRA projector."""
        self._projections = None
        self._astra_proj_link = None
        print("Deleting projector via projections")
        self._astra_projector_id = None

    @property
    def volume_geometry(self) -> VolumeGeometry:
        return self._vol_geom

    @volume_geometry.setter
    def volume_geometry(self, value: VolumeGeometry):
        del self.volume_geometry  # invalidate projector

        if self.volume is not None:
            expected_shape_volume = tuple(reversed(self.volume.shape))
            if value.shape != expected_shape_volume:
                raise ValueError(
                    f"The given volume geometry does not match the "
                    f"shape of the volume in the projector. Got "
                    f"{value.shape} in (x, y, z) convention of the "
                    f"volume geometry, but the volume shape is "
                    f"{self.volume.shape} in (z, y, x) convention.")

        self._vol_geom = value
        vol_shp = value.shape
        ext_min = value.extent_min
        ext_max = value.extent_max
        self._astra_vol_geom = astra.create_vol_geom(
            vol_shp[1], vol_shp[0], vol_shp[2],
            ext_min[1], ext_max[1],
            ext_min[0], ext_max[0],
            ext_min[2], ext_max[2],
        )

    @volume_geometry.deleter
    def volume_geometry(self):
        self._vol_geom = None
        self._astra_vol_geom = None  # TODO: does this delete

        print("Deleting projector via volgeom")
        self._astra_projector_id = None

    @property
    def projection_geometry(self):
        return self._proj_geom

    @projection_geometry.setter
    def projection_geometry(self, value: List[ProjectionGeometry]):
        del self.projection_geometry  # invalidate projector

        if self.projections is not None:
            if self.projections.shape[1] != len(value):
                raise ValueError("Number of projections does not match"
                                 " projection geometry.")
            if (self.projections.shape[0] != value[0].detector.rows
                or self.projections.shape[2] != value[0].detector.cols):
                raise ValueError("Projection shape does not match projection"
                                 " geometry shape.")

        self._proj_geom = value
        det_shape = (self._proj_geom[0].detector.rows,
                     self._proj_geom[0].detector.cols)
        vectors = self._geoms2vectors(value)
        self._astra_proj_geom = astra.create_proj_geom(
            'cone_vec', *det_shape, vectors)

    @projection_geometry.deleter
    def projection_geometry(self):
        self._proj_geom = None
        self._astra_proj_geom = None  # TODO: does this delete

        print("Deleting projector via projgeom")
        self._astra_projector_id = None

    @staticmethod
    def _geoms2vectors(geoms: List[ProjectionGeometry]):
        """Convert a list of geometries to ASTRA vectors."""

        def geom2astra(g):
            c = lambda x: (x[1], x[0], x[2])
            d = lambda x: np.array((x[1], x[0], x[2])) * g.detector.pixel_width
            e = lambda x: np.array(
                (x[1], x[0], x[2])) * g.detector.pixel_height
            return [*c(g.source_position), *c(g.detector_position),
                    *d(g.u), *e(g.v)]

        vectors = np.zeros((len(geoms), 12))
        for i, g in enumerate(geoms):
            vectors[i] = geom2astra(g)
        return vectors

    def projector3d(self):
        if self._astra_projector_id is None:
            if self._astra_proj_geom is None:
                raise RuntimeError("Projection geometries must be given before"
                                   " creating a projector. Use the "
                                   "`projection_geometry` setter of this"
                                   " class.")
            if self._astra_vol_geom is None:
                raise RuntimeError("`VolumeGeometry` must be given before "
                                   "creating a projector. Use the "
                                   "`volume_geometry` setter of this class")
            proj_cfg = {'type': 'cuda3d',
                        'VolumeGeometry': self._astra_vol_geom,
                        'ProjectionGeometry': self._astra_proj_geom,
                        'options': {}}
            print("Setting projector")
            self._astra_projector_id = astra.projector3d.create(proj_cfg)
        return self._astra_projector_id


class AstraCompatConeProjector(CompatProjector):
    def __call__(self, additive=False):
        astra.experimental.direct_FPBP3D(
            self.projector3d(),
            self._astra_vol_link,
            self._astra_proj_link,
            int(not additive),
            "FP")
        return self._projections


class AstraCompatConeBackprojector(CompatProjector):
    def __call__(self, additive=False):
        """
        Backproject a set of projections in self.projections.

        Parameters
        ----------
        volume_extent_min  :   tuple of floats  (z, y, x)
        volume_extent_max  :   tuple of floats  (z, y, x)
        volume_shape    :   tuple of ints    (z, y, x)  (optional)
        volume_axes     :   tuple of ints    (z, y, x)  (optional)
        additive        :   bool    (optional)
        out        :   ndarray (optional)

        Returns
        -------
        out : ndarray   (z, y, x) or (x, y, z) ndarray  (optional)
        """
        astra.experimental.direct_FPBP3D(
            self.projector3d(),
            self._astra_vol_link,
            self._astra_proj_link,
            int(not additive),
            "BP")
        return self._volume
