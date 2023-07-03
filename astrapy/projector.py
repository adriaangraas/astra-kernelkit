from typing import Any, Callable, List, Optional, Tuple
import cupy as cp
from abc import ABC, abstractmethod

from astrapy.data import aspitched, ispitched
from astrapy.geom.proj import GeometrySequence, ProjectionGeometry
from astrapy.geom.vol import VolumeGeometry
from astrapy.kernel import copy_to_texture
from astrapy.kernels.cone_bp import ConeBackprojection
from astrapy.kernels.cone_fp import ConeProjection


class BaseProjector(ABC):
    """Base class for projectors.

    These classes are optimized for reusing geometries and memory for typical
    reconstruction problems, where there is a single set of projections and a
    single reconstruction volume."""

    @property
    @abstractmethod
    def projection_geometry(self) -> Optional[List[ProjectionGeometry]]:
        raise NotImplementedError()

    @projection_geometry.setter
    @abstractmethod
    def projection_geometry(self, value: List[ProjectionGeometry]):
        raise NotImplementedError()

    @projection_geometry.deleter
    @abstractmethod
    def projection_geometry(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def volume_geometry(self) -> Optional[VolumeGeometry]:
        raise NotImplementedError()

    @volume_geometry.setter
    @abstractmethod
    def volume_geometry(self, value: VolumeGeometry):
        raise NotImplementedError()

    @volume_geometry.deleter
    @abstractmethod
    def volume_geometry(self):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, additive: bool = False):
        raise NotImplementedError()


class ConeBaseProjector(BaseProjector, ABC):
    """Base class for cone-beam projectors."""

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

    @volume_geometry.deleter
    def volume_geometry(self):
        del self._vol_geom


class ConeProjector(ConeBaseProjector):
    """Cone-beam forward projector."""

    def __init__(self, kernel: ConeProjection = None,
                 volume_axes=(0, 1, 2), projection_axes=(0, 1, 2)):
        """Constructor.

        Parameters
        ----------
        kernel : ConeProjection, optional
            The kernel to use for projection. If `None`, the default kernel is
            used.
        volume_axes : tuple, optional
            The axes of the volume array that correspond to the x, y, and z
            axes of the volume geometry. The default is (0, 1, 2).
        projection_axes : tuple, optional
            The axes of the projection array or list that correspond to the
            angles, u, and v axes of the projection geometry. The default is
            (0, 1, 2).
        verbose : bool, optional
            Whether to print verbose output. The default is True.
        """
        super().__init__()
        self._K = ConeProjection(
            volume_axes=volume_axes,
            projection_axes=projection_axes) if kernel is None else kernel
        self._vol = None
        self._projs = None
        self._vol_txt = None
        self._vol_txt_cuda_array_valid = True
        self._graph: Optional[cp.cuda.Graph] = None

    @property
    def volume(self) -> Optional[cp.ndarray]:
        return self._vol

    @volume.setter
    def volume(self, value):
        """Sets or updates the reconstruction volume in place.

        Parameters
        ----------
        value : array-like
            The reconstruction volume. Must be a 3D array with the same shape
            as the volume geometry."""
        if self._vol is not None and value.shape == self._vol.shape:
            self._vol[...] = value
            self._vol_txt_cuda_array_valid = False
        else:
            self._vol = value

    @volume.deleter
    def volume(self):
        """Deletes the reconstruction volume and frees texture memory."""
        self._vol = None
        del self._vol_txt

    def _compute_texture(self):
        """Computes the texture for the reconstruction volume. If
        `self._vol_txt` is set, the texture is updated in place."""
        if self._vol_txt is None:
            self._vol_txt = copy_to_texture(self._vol)
        elif not self._vol_txt_cuda_array_valid:
            self._vol_txt.ResDesc.cuArr.copy_from(self._vol)

        self._vol_txt_cuda_array_valid = True
        return self._vol_txt

    @property
    def projections(self) -> Optional[cp.ndarray]:
        return self._projs

    @projections.setter
    def projections(self, value: cp.ndarray):
        # TODO
        self._projs = value

    def __call__(self, additive=False):
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
        txt = self._compute_texture()
        self._K(txt,
                self._vol_geom,
                self._proj_geom_seq,
                self._projs)
        return self._projs


class ConeBackprojector(ConeBaseProjector):
    """Backprojector for cone-beam geometry"""

    TEXTURE_TYPES = ('array', 'pitch2d')

    def __init__(self, kernel: ConeBackprojection = None,
                 texture_type: str = 'array',
                 volume_axes=(0, 1, 2), projection_axes=(0, 1, 2)):
        """
        Constructor.

        Parameters
        ----------
        kernel : ConeBackprojection, optional
            The kernel to use for backprojection. If `None`, the default kernel
            is used.
        texture_type : str, optional
            The type of texture to use for the projection data. Can be either
            'array' or 'pitch2d'. The default is 'array'. If 'pitch2d' is
            chosen, the projection data is copied into a pitched GPU array on
            first arrival. Creation of 'pitch2d' textures is faster, but usage
            of 'array' textures is faster.
        volume_axes : tuple, optional
            The axes of the volume array that correspond to the x, y, and z
            axes of the volume geometry. The default is (0, 1, 2). See also
            the `ConeBackprojection` kernel.
        projection_axes : tuple, optional
            The axes of the projection array or list that correspond to the
            angles, u, and v axes of the projection geometry. The default is
            (0, 1, 2). See also the `ConeBackprojection` kernel.
        """
        super().__init__()
        self._K = ConeBackprojection(
            # limit_projs_per_block=256,
            volume_axes=volume_axes,
            projection_axes=projection_axes
        ) if kernel is None else kernel
        self._texture_type = texture_type.lower()
        if self._texture_type not in self.TEXTURE_TYPES:
            raise ValueError(f"`texture_type` not supported. Please choose "
                             f"one of the following: {self.TEXTURE_TYPES}.")
        self._textures = None
        self._texture_cuda_array_valid = True
        self._texture_2d_objects = None
        self._params = None
        self._projections = None
        self._graph: Optional[cp.cuda.Graph] = None
        self._vol = None
        self._tmp = None  # TODO: avoid using a tmp while still enabling graphs
        self._vol_axs = volume_axes
        self._proj_axs = projection_axes

    @property
    def volume(self) -> Optional[cp.ndarray]:
        """Returns the reconstruction volume."""
        return self._vol

    @volume.setter
    def volume(self, value):
        """Sets or updates the reconstruction volume in place."""
        if self.volume_geometry is not None:
            expected_shp = (self._vol_geom.shape[i] for i in self._vol_axs)
            if value.shape != tuple(expected_shp):
                raise ValueError(
                    f"Volume shape {value.shape} does not match volume "
                    f"geometry {self._vol_geom.shape} with axes "
                    f"{self._vol_axs}.")
        self._vol = value
        if self._tmp is None:  # TODO
            self._tmp = cp.empty_like(value)

    @property
    def projections(self):
        """Returns the projection data. Note the array may be pitched if
        `texture_type` is 'pitch2d'."""
        return self._projections

    @projections.setter
    def projections(self, value):
        """Sets or updates the projection data in place. The array is converted
        to a pitched texture if `texture_type` is 'pitch2d'. If the texture
        type is 'array', this maintains the texture memory, but invalidates it,
        triggering a new conversion on th next __call__."""
        if self.projection_geometry is not None:
            shp = (len(self.projection_geometry),
                   self.projection_geometry[0].detector.rows,
                   self.projection_geometry[0].detector.cols)
            expected_shp = (shp[i] for i in self._proj_axs)
            if len(value) != tuple(expected_shp)[0]:
                raise ValueError(
                    f"Projections shape {value.shape} does not match "
                    f"projection geometry {shp} with axes {self._proj_axs}.")
        if self._projections is not None:
            if len(self._projections) == len(value):
                for i, (p, v) in enumerate(zip(self._projections, value)):
                    if i == 0:  # TODO: check if asserting every shape is slow
                        assert p.shape == v.shape
                    p[...] = v
        else:
            if self._texture_type == 'pitch2d':
                self._projections = [aspitched(v, xp=cp) for v in value]
            else:
                self._projections = value
        if self._texture_type == 'array':
            self._texture_cuda_array_valid = False  # invalidate CUDA array

    @projections.deleter
    def projections(self):
        """Deletes the projection data as well as the textures."""
        self._projections = None
        if self._texture_type == 'array':
            self._textures = None  # invalidate textures if CUDA array
        self._texture_2d_objects = None

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
                assert [ispitched(p) for p in self._projections]
                # needs to maintain in memory
                self._texture_2d_objects = [
                    copy_to_texture(p, self._texture_type)
                    for p in self._projections]
                txt = cp.array([p.ptr for p in self._texture_2d_objects])
            else:
                # TODO: avoid the `asarray` here, let the caller do it
                projs = cp.asarray(self.projections)
                txt = copy_to_texture(projs, self._texture_type)
            self._textures = txt
        elif self._texture_type == 'array' and not self._texture_cuda_array_valid:
            # TODO: avoid the `asarray` here, let the caller do it
            self._textures.ResDesc.cuArr.copy_from(
                cp.asarray(self._projections))
            self._texture_cuda_array_valid = True
        return self._textures

    def __call__(self, additive=False):
        """Backprojects the projections into the volume.

        A first call to this method will capture a CUDA graph, which will be
        launched on subsequent calls. This is only possible if the CUDA graph
        capture mode is enabled (see `cupy.cuda.set_graph_capture_mode`).

        The current implementation only supports output volumes of axes
        (0, 1, 2) or (0, 2, 1). The reason is that, for coalesced writes in
        global memory, the fastest varying dimension should be the last one.
        For different geometries, the kernel can be parametrized with Jinja2,
        but this is still work in progress.

        Parameters
        ----------
        additive : bool, optional
            If True, the volume is updated, rather than overwitten.

        Returns
        -------
        volume : array-like
            The reconstructed volume.
        """

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

        # txts = self._compute_textures()
        # params = self._compute_params()
        # if not additive:
        #     self.volume.fill(0.)
        # self._K(txts, params, self._vol, self.volume_geometry)

        with cp.cuda.stream.Stream() as stream:
            self._tmp.fill(0.)
            txts = self._compute_textures()
            params = self._compute_params()
            if self._graph is not None:
                self._graph.launch()
            else:
                if stream != stream.null:
                    stream.begin_capture()
                self._K(txts, params, self._tmp, self.volume_geometry)
                if stream.is_capturing():
                    self._graph = stream.end_capture()
                    self._graph.launch()
            if not additive:
                self._vol[...] = self._tmp
            else:
                self._vol += self._tmp

        if self._vol_axs == (2, 1, 0):
            pass
        elif self._vol_axs == (0, 1, 2):
            # returning a view (an array transpose)
            # TODO(Adriaan): think about a more sustainable solution
            self._vol[...] = cp.reshape(
                self._vol,
                tuple(reversed(self._vol.shape))).T
        else:
            raise NotImplementedError("Sorry! Not yet implemented, but should"
                                      " be easy to do so. Please open an issue.")
        return self._vol
