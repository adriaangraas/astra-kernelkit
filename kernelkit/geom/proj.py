import copy
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence
import numpy as np
import transforms3d


class Beam(Enum):
    """Beam type."""
    CONE = 'cone'
    PARALLEL = 'parallel'


@dataclass
class Detector:
    """Detector geometry."""

    rows: int
    cols: int
    pixel_height: float
    pixel_width: float

    @property
    def width(self):
        return self.pixel_width * self.cols

    @property
    def height(self):
        return self.pixel_height * self.rows

    @property
    def pixel_volume(self):
        return self.pixel_width * self.pixel_height


class ProjectionGeometry:
    """Geometry of a single projection."""
    ANGLES_CONVENTION = "sxyz"

    def __init__(self,
                 source_position: Sequence,
                 detector_position: Sequence,
                 u_unit: Sequence,
                 v_unit: Sequence,
                 detector: Detector,
                 beam: str = Beam.CONE):
        """Initialize a projection geometry.

        Parameters
        ----------
        source_position : Sequence
            Position of the source in the world frame.
        detector_position : Sequence
            Position of the detector in the world frame.
        u_unit : Sequence
            Vector pointing in the horizontal direction of the detector of
            length 1.
        v_unit : Sequence
            Vector pointing in the vertical direction of the detector of
            length 1.
        detector : Detector
            Detector object, see :class:`astrapy.geom.Detector`.
        beam : str
            Beam type, either 'cone' or 'parallel'.
        """
        self.source_position = np.array(source_position)
        self.detector_position = np.array(detector_position)
        self.detector = detector
        self.u = np.array(u_unit)
        self.v = np.array(v_unit)

        if beam not in Beam:
            raise ValueError(f"Beam type must be one of {Beam}.")
        self.beam = beam

    @property
    def detector_extent_min(self):
        """Minimum extent of the detector in the world frame.
        The detector is assumed to be centered around the detector position."""
        return (self.detector_position
                - self.v * self.detector.height / 2
                - self.u * self.detector.width / 2)

    @staticmethod
    def angles2mat(r, p, y) -> np.ndarray:
        """Convert roll, pitch, yaw angles to a rotation matrix."""
        return transforms3d.euler.euler2mat(
            r, p, y, ProjectionGeometry.ANGLES_CONVENTION)

    @staticmethod
    def mat2angles(mat) -> float:
        """Convert a rotation matrix to roll, pitch, yaw angles."""
        return transforms3d.euler.mat2euler(
            mat, ProjectionGeometry.ANGLES_CONVENTION)

    @property
    def u(self):
        """Horizontal u-vector in the detector frame of unit length."""
        return self.__u

    @u.setter
    def u(self, value):
        """Set the horizontal u-vector in the detector frame.

        Parameters
        ----------
        value : array-like
            The horizontal u-vector in the detector frame. Must be a unit
            vector."""
        if not np.allclose(np.linalg.norm(value), 1.):
            raise ValueError("`u` must be a unit vector.")
        self.__u = value

    @property
    def v(self):
        """Vertical v-vector in the detector frame of unit length.

        Returns
        -------
        array-like
        """
        return self.__v

    @v.setter
    def v(self, value):
        """Set the vertical v-vector in the detector frame.

        Parameters
        ----------
        value : array-like
            The vertical v-vector in the detector frame. Must be a unit
            vector."""
        if not np.allclose(np.linalg.norm(value), 1.):
            raise ValueError("`v` must be a unit vector.")
        self.__v = value

    def __str__(self):
        return (f"Source {self.source_position} "
                f"Detector {self.detector_position}"
                f"U {self.u}"
                f"V {self.v}")


@dataclass
class GeometrySequence:
    """Structure-of-arrays geometry data object

    Parameters
    ----------
    source_position : array-like
        The position of the source in the world frame.
    detector_position : array-like
        The position of the detector in the world frame.
    u : array-like
        The horizontal u-vector in the detector frame.
    v : array-like
        The vertical v-vector in the detector frame.
    detector : DetectorSequence
        The detector data.
    xp : array-like, optional
        The array library to use. If None, it defaults to numpy.
    """
    xp = np

    @dataclass
    class DetectorSequence:
        """Structure-of-arrays detector data object

        Parameters
        ----------
        rows : array-like
            The number of rows in each detector.
        cols : array-like
            The number of columns in each detector.
        pixel_width : array-like
            The width of each pixel in each detector.
        pixel_height : array-like
            The height of each pixel in each detector.
        xp : array-like, optional
            The array library to use. If None, it defaults to numpy.
        """
        xp = np
        rows: xp.ndarray
        cols: xp.ndarray
        pixel_width: xp.ndarray
        pixel_height: xp.ndarray

        @property
        def height(self):
            arr = self.xp.empty_like(self.pixel_height)  # preserve dtype
            self.xp.multiply(self.pixel_height, self.rows, out=arr)
            return arr

        @property
        def width(self):
            arr = self.xp.empty_like(self.pixel_width)  # preserve dtype
            self.xp.multiply(self.pixel_width, self.cols, out=arr)
            return arr

        @property
        def pixel_volume(self):
            return self.pixel_width * self.pixel_height

    source_position: xp.ndarray
    detector_position: xp.ndarray
    u: xp.ndarray
    v: xp.ndarray
    detector: DetectorSequence

    def __len__(self):
        return len(self.source_position)

    @property
    def detector_extent_min(self):
        """The minimum extent of the detector in the detector frame.
        This is the bottom left corner of the detector."""
        return (self.detector_position
                - self.v * self.detector.height[..., self.xp.newaxis] / 2
                - self.u * self.detector.width[..., self.xp.newaxis] / 2)

    @classmethod
    def fromList(cls, geometries: List[ProjectionGeometry], xp=None):
        """Create a GeometrySequence from a list of ProjectionGeometry objects.

        Parameters
        ----------
        geometries : List[ProjectionGeometry]
            A list of ProjectionGeometry objects.
        xp : array-like, optional
            The array library to use. If None, it defaults to numpy.
        """
        if xp is None:
            xp = cls.xp

        def _cvrt(arr, dtype): return xp.ascontiguousarray(
            xp.array(arr, dtype=dtype))

        ds = cls.DetectorSequence(
            rows=_cvrt([g.detector.rows for g in geometries], xp.int32),
            cols=_cvrt([g.detector.cols for g in geometries], xp.int32),
            pixel_width=_cvrt([g.detector.pixel_width for g in geometries],
                              xp.float32),
            pixel_height=_cvrt([g.detector.pixel_height for g in geometries],
                               xp.float32))
        gs = cls(
            source_position=_cvrt([g.source_position for g in geometries],
                                  xp.float32),
            detector_position=_cvrt([g.detector_position for g in geometries],
                                    xp.float32),
            u=_cvrt([g.u for g in geometries], xp.float32),
            v=_cvrt([g.v for g in geometries], xp.float32),
            detector=ds)
        return gs

    def take(self, indices):
        """Take a subset of the geometry sequence.

        Parameters
        ----------
        indices : array-like
            Indices to take.
        """
        ds = self.DetectorSequence(
            rows=self.detector.rows[indices],
            cols=self.detector.cols[indices],
            pixel_width=self.detector.pixel_width[indices],
            pixel_height=self.detector.pixel_height[indices]
        )
        gs = GeometrySequence(
            source_position=self.source_position[indices],
            detector_position=self.detector_position[indices],
            u=self.u[indices],
            v=self.v[indices],
            detector=ds)
        return gs

    def __getitem__(self, item):
        return self.take(item)

    def __deepcopy__(self, memodict={}):
        xp = self.xp
        obj = self.__class__(
            source_position=xp.copy(self.source_position),
            detector_position=xp.copy(self.detector_position),
            u=xp.copy(self.u),
            v=xp.copy(self.v),
            detector=GeometrySequence.DetectorSequence(
                rows=xp.copy(self.detector.rows),
                cols=xp.copy(self.detector.cols),
                pixel_width=xp.copy(self.detector.pixel_width),
                pixel_height=xp.copy(self.detector.pixel_height),
            )
        )
        obj.xp = xp
        obj.detector.xp = xp
        return obj


def shift(geom: ProjectionGeometry,
          shift_vector: np.ndarray) -> ProjectionGeometry:
    """Creates a new geometry by shifting an existing one in 3D space

    Parameters
    ----------
    geom: ProjectionGeometry or GeometrySequence
    shift_vector: numpy.ndarray, cupy.ndarray, shape (3,), or (N, 3)
    for GeometrySequence

    Returns
    -------
    New `ProjectionGeometry` or `GeometrySequence`
    """
    geom = copy.deepcopy(geom)
    geom.source_position += shift_vector
    geom.detector_position += shift_vector
    return geom


def shift_(geom: ProjectionGeometry, shift_vector: np.ndarray) -> None:
    """In-place shift of a geometry or geometry sequence

    Parameters
    ----------
    geom: ProjectionGeometry or GeometrySequence
    shift_vector: numpy.ndarray, cupy.ndarray, shape (3,), or (N, 3)
    for GeometrySequence
    """
    geom.source_position += shift_vector
    geom.detector_position += shift_vector


def scale(geom: ProjectionGeometry, scaling: np.ndarray) -> ProjectionGeometry:
    """Creates a new geometry by scaling an existing one

    Parameters
    ----------
    geom : ProjectionGeometry or GeometrySequence
    scaling : numpy.ndarray, cupy.ndarray, shape (3,), or (N, 3)
    for GeometrySequence

    Returns
    -------
    New `ProjectionGeometry` or `GeometrySequence`
    """

    geom = copy.deepcopy(geom)
    # detector pixels have to be scaled first, because
    # detector.width and detector.height need to be scaled accordingly
    horiz_pixel_vector = (geom.u * geom.detector.pixel_width) / scaling
    new_pixel_width = np.linalg.norm(horiz_pixel_vector)
    new_u_unit = horiz_pixel_vector / new_pixel_width
    geom.detector.pixel_width = new_pixel_width
    geom.u = new_u_unit

    vert_pixel_vector = (geom.v * geom.detector.pixel_height) / scaling
    new_pixel_height = np.linalg.norm(vert_pixel_vector)
    new_v_unit = vert_pixel_vector / new_pixel_height
    geom.detector.pixel_height = new_pixel_height
    geom.v = new_v_unit

    geom.source_position[:] = geom.source_position[:] / scaling
    geom.detector_position[:] = geom.detector_position[:] / scaling
    return geom


def scale_(geom: ProjectionGeometry, scaling: float) -> None:
    """In-place scaling of geometry or geometry sequence

    Parameters
    ----------
    geom : ProjectionGeometry or GeometrySequence
    scaling : float
        Factor by which to enlarge or shrink the geometry
    """
    # detector pixels have to be scaled first, because
    # detector.width and detector.height need to be scaled accordingly
    xp = geom.xp
    pixel_vec = geom.u * geom.detector.pixel_width[..., xp.newaxis]
    pixel_vec /= scaling
    geom.detector.pixel_width = xp.linalg.norm(pixel_vec, axis=-1)
    xp.divide(pixel_vec,
              geom.detector.pixel_width[..., xp.newaxis],
              out=geom.u)
    xp.multiply(geom.v, geom.detector.pixel_height[..., xp.newaxis],
                out=pixel_vec)
    pixel_vec /= scaling
    geom.detector.pixel_height = xp.linalg.norm(pixel_vec, axis=-1)
    xp.divide(pixel_vec,
              geom.detector.pixel_height[..., xp.newaxis],
              out=geom.v)
    xp.divide(geom.source_position, scaling, out=geom.source_position)
    xp.divide(geom.detector_position, scaling, out=geom.detector_position)


def rotate(geom: ProjectionGeometry,
           roll: float = 0.,
           pitch: float = 0.,
           yaw: float = 0.) -> ProjectionGeometry:
    """Creates a new geometry by rotating an existing

    Parameters
    ----------
    geom : ProjectionGeometry or GeometrySequence
    roll : float
        Rotation around x-axis
    pitch : float
        Rotation around y-axis
    yaw : float
        Rotation around z-axis

    Returns
    -------
    New `ProjectionGeometry` or `GeometrySequence`
    """
    ngeom = copy.deepcopy(geom)
    RT = ProjectionGeometry.angles2mat(roll, pitch, yaw).T
    ngeom.source_position = RT @ geom.source_position
    ngeom.detector_position = RT @ geom.detector_position
    ngeom.u = RT @ geom.u
    ngeom.v = RT @ geom.v
    return ngeom


def rotate_(geom: ProjectionGeometry,
            roll: float = 0., pitch: float = 0., yaw: float = 0.) -> None:
    """In-place rotation of geometry

    Parameters
    ----------
    geom : ProjectionGeometry or GeometrySequence
    roll : float
        Rotation around x-axis
    pitch : float
        Rotation around y-axis
    yaw : float
        Rotation around z-axis
    """
    R = geom.xp.asarray(ProjectionGeometry.angles2mat(roll, pitch, yaw),
                        dtype=geom.source_position.dtype)
    geom.source_position[...] = geom.source_position @ R
    geom.detector_position[...] = geom.detector_position @ R
    geom.u[...] = geom.u @ R
    geom.v[...] = geom.v @ R
