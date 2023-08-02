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
    def projection_axes(self):
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

    @property
    @abstractmethod
    def volume_axes(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def volume(self) -> Optional[cp.ndarray]:
        raise NotImplementedError()

    @volume.setter
    @abstractmethod
    def volume(self, value: cp.ndarray):
        raise NotImplementedError()

    @volume.deleter
    @abstractmethod
    def volume(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def projections(self) -> Optional[cp.ndarray]:
        raise NotImplementedError()

    @projections.setter
    @abstractmethod
    def projections(self, value: cp.ndarray):
        raise NotImplementedError()

    @projections.deleter
    @abstractmethod
    def projections(self):
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, additive: bool = False):
        raise NotImplementedError()


class ConeBaseProjector(BaseProjector, ABC):
    """Base class for cone-beam projectors."""

    def __init__(self, kernel):
        self._vol_geom: Optional[VolumeGeometry] = None
        self._proj_geom: Optional[List[ProjectionGeometry]] = None
        self._proj_geom_seq = None
        self._K = kernel

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
    def projection_axes(self):
        return self._K.projection_axes

    @property
    def volume_geometry(self) -> Optional[VolumeGeometry]:
        return self._vol_geom

    @volume_geometry.setter
    def volume_geometry(self, value: VolumeGeometry):
        self._vol_geom = value

    @volume_geometry.deleter
    def volume_geometry(self):
        del self._vol_geom

    @property
    def volume_axes(self):
        return self._K.volume_axes


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
        K = ConeProjection(
            volume_axes=volume_axes,
            projection_axes=projection_axes) if kernel is None else kernel
        super().__init__(K)
        self._vol_txt = None
        self._vol_txt_cuda_array_valid = True
        self._graph: Optional[cp.cuda.Graph] = None

    @property
    def volume(self) -> Optional[cp.ndarray]:
        try:
            return self._vol
        except AttributeError:
            raise AttributeError("Volume not set. Use the `volume` setter "
                                 "before calling the projector.")


    @volume.setter
    def volume(self, value):
        """Sets or updates the reconstruction volume in place.

        Parameters
        ----------
        value : array-like
            The reconstruction volume. Must be a 3D array with the same shape
            as the volume geometry."""
        if hasattr(self, '_vol') and value.shape == self._vol.shape:
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
        try:
            return self._projs
        except AttributeError:
            raise AttributeError("Projections not set. Use the `projections` "
                                 "setter before calling the projector.")

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

    TEXTURE_TYPES = ('auto', 'array', 'layered', 'array2d', 'pitch2d')

    def __init__(self, kernel: ConeBackprojection = None,
                 texture_type: str = 'auto',
                 use_graphs: bool = True,
                 **kwargs):
        """
        Constructor.

        Parameters
        ----------
        kernel : ConeBackprojection, optional
            The kernel to use for backprojection. If `None`, the default kernel
            is used.
        texture_type : str, optional
            The type of texture to use for the projection data. Can be either
            'array' or 'pitch2d'. The default is 'auto'. If 'pitch2d' is
            chosen, the projection data is copied into a pitched GPU array on
            first arrival. Creation of 'pitch2d' textures is faster, but usage
            of 'array' textures is faster. If 'auto', it is 'pitch2d' if a list
            or pitched arrays is given, 'array2d' if a list of arrays is given,
            'layered' if a 3D cupy array with first axis is the angle axis,
            and 'array' otherwise.

        """
        K = ConeBackprojection(**kwargs) if kernel is None else kernel
        if kernel is not None and len(kwargs) > 0:
            raise ValueError(
                "If `kernel` is given, other keyword arguments are "
                "not passed through the kernel.")
        super().__init__(K)
        self._texture_type = texture_type.lower()
        if self._texture_type not in self.TEXTURE_TYPES:
            raise ValueError(f"`texture_type` not supported. Please choose "
                             f"one of the following: {self.TEXTURE_TYPES}.")
        self._texture_cuda_array_valid = True
        self._use_graphs = use_graphs

    @property
    def volume(self) -> Optional[cp.ndarray]:
        """Returns the reconstruction volume."""
        try:
            return self._vol
        except AttributeError:
            raise AttributeError("No volume set. Please set the volume "
                                 "before calling the backprojector.")

    @volume.setter
    def volume(self, value):
        """Sets or updates the reconstruction volume in place."""
        if self.volume_geometry is not None:
            expected_shp = (self._vol_geom.shape[i] for i in
                            self._K.volume_axes)
            if value.shape != tuple(expected_shp):
                raise ValueError(
                    f"Volume shape {value.shape} does not match volume "
                    f"geometry {self._vol_geom.shape} with axes "
                    f"{self._K.volume_axes}.")
        self._vol = value

    @volume.deleter
    def volume(self):
        """Deletes the reconstruction volume and frees texture memory."""
        self._vol = None

    @property
    def projections(self):
        """Returns the projection data. Note the array may be pitched if
        `texture_type` is 'pitch2d'."""
        try:
            return self._projections
        except AttributeError:
            raise AttributeError("No projections set. Please set the "
                                 "projections before calling the projector.")

    @projections.setter
    def projections(self, value: Any):
        """Sets or updates the projection data in place. The array is converted
        to a pitched texture if `texture_type` is 'pitch2d'. If the texture
        type is 'array', this maintains the texture memory, but invalidates it,
        triggering a new conversion on th next __call__.

        Parameters
        ----------
        value : cp.ndarray
            The projection data."""
        if not isinstance(value, cp.ndarray) and not isinstance(value, List):
            raise ValueError("Projections must be a cupy array or list.")

        # if geometry given, check if it matches
        if self.projection_geometry is not None:
            shp = (len(self.projection_geometry),
                   self.projection_geometry[0].detector.rows,
                   self.projection_geometry[0].detector.cols)
            expected_shp = tuple((shp[i] for i in self._K.projection_axes))
            if isinstance(value, List):
                if (len(value) != expected_shp[0]):
                    raise ValueError(
                        f"Projections shape {len(value)} does not match "
                        f"projection geometry {shp} with axes {self._K.projection_axes}.")
                if value[0].shape != expected_shp[1:]:
                    raise ValueError(
                        f"Projections shape {value[0].shape} does not match "
                        f"projection geometry {shp} with axes {self._K.projection_axes}.")
            elif isinstance(value, cp.ndarray):
                if value.shape != expected_shp:
                    raise ValueError(
                        f"Projections shape {value.shape} does not match "
                        f"projection geometry {shp} with axes {self._K.projection_axes}.")

        # if projections already exist, update in-place
        if hasattr(self, '_projections'):
            if len(self._projections) == len(value):
                for i, (p, v) in enumerate(zip(self._projections, value)):
                    if i == 0:  # TODO: check if asserting every shape is slow
                        assert p.shape == v.shape
                    p[...] = v  # update, works also in 3D
        else:
            if self._texture_type == 'auto':  # find out texture type here
                if isinstance(value, List):
                    if ispitched(value[0]):
                        self._texture_type = 'pitch2d'
                    else:
                        self._texture_type = 'array2d'
                else:
                    if self.projection_axes[0] == 0:
                        self._texture_type = 'layered'
                    else:
                        self._texture_type = 'array'
                print(f"Using texture type {self._texture_type}.")

            if isinstance(value, List):
                if self._K.projection_axes[0] != 0:
                    raise ValueError("Can only use list of projections when "
                                     "projection axis is 0.")

            if self._texture_type == 'pitch2d':
                self._projections = [aspitched(v, xp=cp) for v in value]
            else:
                if self._texture_type == 'layered':
                    if self._K.projection_axes[0] != 0:
                        raise ValueError("Can only use layered textures when "
                                         "the first projection axis is 0.")
                self._projections = value

        # invalidate CUDA arrays, but keep allocated
        if self._texture_type in ('array', 'array2d'):
            self._texture_cuda_array_valid = False  # invalidate CUDA array

    @projections.deleter
    def projections(self):
        """Deletes the projection data as well as the textures."""
        del self._projections
        del self._textures
        del self._texture_2d_objects

    def _compute_params(self):
        """Compute the parameters for the backprojection kernel."""

        if not hasattr(self, '_params'):
            if self._proj_geom_seq is None:
                raise RuntimeError("Make sure to set the `geometry` property"
                                   " of this projector before calling it.")
            self._params = self._K.geoms2params(self._proj_geom_seq,
                                                self.volume_geometry)
        return self._params

    def _compute_textures(self):
        """Compute the textures for the backprojection kernel."""

        # set textures to a single texture object or a cupy array of pointers
        if not hasattr(self, '_textures'):
            if not hasattr(self, '_projections'):
                raise RuntimeError(
                    "Make sure to set the `projections` property"
                    " of this projector before calling it.")
            if self._texture_type in ('pitch2d', 'array2d'):
                if self._texture_type == 'pitch2d':
                    assert [ispitched(p) for p in self._projections]
                # avoid garbage collection
                self._texture_2d_objects = [
                    copy_to_texture(p, self._texture_type)
                    for p in self._projections]
                txt = cp.array([p.ptr for p in self._texture_2d_objects])
            elif self._texture_type in ('array', 'layered'):
                use_layered = self._texture_type == 'layered'
                txt = copy_to_texture(self._projections, 'array',
                                      layered=use_layered)
            else:
                raise ValueError("Unknown texture type.")

            self._textures = txt
        else:  # update CUDA arrays if necessary
            if (self._texture_type in ('array', 'array2d', 'layered')
                and not self._texture_cuda_array_valid):
                if isinstance(self._projections, List):  # array2d
                    for p, t in zip(self._projections,
                                    self._texture_2d_objects):
                        t.ResDesc.cuArr.copy_from(p)
                else:  # array or layered
                    assert isinstance(self._projections, cp.ndarray)
                    assert self._projections.ndim == 3
                    self._textures.ResDesc.cuArr.copy_from(self._projections)
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
        stream = cp.cuda.get_current_stream()

        if not additive:
            self.volume.fill(0.)

        txts = self._compute_textures()
        params = self._compute_params()
        if self._use_graphs and hasattr(self, '_graph'):
            self._graph.launch()
            print('.', end='')
        else:
            if self._use_graphs and stream != stream.null:
                stream.begin_capture()
            self._K(txts, params, self._vol, self.volume_geometry)
            if stream.is_capturing():
                self._graph = stream.end_capture()
                self._graph.launch()

        if self._K.volume_axes == (2, 1, 0):
            pass
        elif self._K.volume_axes == (0, 1, 2):
            # returning a view (an array transpose)
            # TODO(Adriaan): think about a more sustainable solution
            self._vol[...] = cp.reshape(
                self._vol,
                tuple(reversed(self._vol.shape))).T
        # TODO(Adriaan): quickly disabled for kernel tuner:
        # else:
        #     raise NotImplementedError(
        #         "Sorry! Not yet implemented, but should"
        #         " be easy enough to do so. Please open an issue.")

    def clear(self):
        """Clear cached textures and CUDA graph."""
        try:
            del self._textures
            del self._texture_2d_objects
            del self._graph
        except AttributeError:
            pass
        self._texture_cuda_array_valid = False