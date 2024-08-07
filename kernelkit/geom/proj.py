import copy
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np
import transforms3d

ANGLES_CONVENTION = "sxyz"  # using the transforms3d specification


def angles2mat(r, p, y) -> np.ndarray:
    """Convert roll, pitch, yaw angles to a rotation matrix.

    Parameters
    ----------
    r : float
        Roll angle in radians.
    p : float
        Pitch angle in radians.
    y : float
        Yaw angle in radians.

    Returns
    -------
    np.ndarray
        Rotation matrix.
    """
    return transforms3d.euler.euler2mat(r, p, y, ANGLES_CONVENTION)


@dataclass
class Detector:
    """Detector specifications.

    Parameters
    ----------
    rows : int
        Number of rows in the detector.
    cols : int
        Number of columns in the detector.
    pixel_height : float
        Height of a single pixel.
    pixel_width : float
        Width of a single pixel.
    """

    __slots__ = "rows", "cols", "pixel_height", "pixel_width"

    rows: int
    cols: int
    pixel_height: float
    pixel_width: float

    @property
    def shape(self):
        return self.rows, self.cols

    @property
    def width(self):
        """Width of the detector."""
        return self.pixel_width * self.cols

    @property
    def height(self):
        """Height of the detector."""
        return self.pixel_height * self.rows

    @property
    def pixel_volume(self):
        """Volume of a single detector pixel."""
        return self.pixel_width * self.pixel_height

    __dict__ = {}


class Beam(Enum):
    """Beam type."""

    CONE = "cone"
    PARALLEL = "parallel"
    
    @classmethod
    def from_string(cls, value: str):
        """Convert a string to a Beam enum value."""
        value = value.lower()
        for beam in cls:
            if beam.value == value:
                return beam
        raise ValueError(f"Invalid beam type: '{value}'")


def process_beam_type(arg) -> Beam:
    if isinstance(arg, Beam):
        return arg
    elif isinstance(arg, str):
        return Beam.from_string(arg)
    else:
        raise TypeError("Argument must be of type `str` or `Beam`.")


class ProjectionGeometry:
    """Single 3D projection with one source and one detector."""

    xp = np  # using CuPy is possible, but not tested

    def __init__(
        self,
        source_position,
        detector_position,
        detector: Detector,
        u: Sequence = (0.0, 1.0, 0.0),
        v: Sequence = (0.0, 0.0, 1.0),
        beam: Beam | str = Beam.CONE,
        array_module=None,
    ):
        """Initialize a projection geometry.

        Parameters
        ----------
        source_position : array-like
            Position of the source in the world frame.
        detector_position : array-like
            Position of the detector in the world frame. The detector is
            assumed to be centered around the detector position.
        detector : kernelkit.geom.Detector
            Detector object.
        u: Sequence
            Vector pointing in the horizontal direction of the detector of
            length 1.
        v: Sequence
            Vector pointing in the vertical direction of the detector of
            length 1.
        beam : str
            Beam type, either 'cone' or 'parallel'.

        Notes
        -----
        The axes conventions always follow the coordinates (x, y, z) in world
        space, where x and y are horizontal, and z is vertical. The detector
        u vector is horizontal and the v vector is vertical.
        """
        if array_module is not None:
            self.xp = array_module
        self.source_position = self.xp.array(source_position, dtype=self.xp.float32)
        self.detector_position = self.xp.array(detector_position, dtype=self.xp.float32)
        self.detector = detector
        self.u = self.xp.array(u, dtype=self.xp.float32)
        self.v = self.xp.array(v, dtype=self.xp.float32)
        self.beam = process_beam_type(beam)

    @property
    def detector_extent_min(self):
        """Minimum extent of the detector in the world frame.

        Notes
        -----
        The detector is assumed to be centered around the detector position.
        """
        return (
            self.detector_position
            - self.v * self.detector.height / 2
            - self.u * self.detector.width / 2
        )

    @property
    def u(self):
        """Horizontal u-vector in the detector frame of unit length."""
        return self._u

    @u.setter
    def u(self, value):
        """Set the horizontal u-vector in the detector frame.

        Parameters
        ----------
        value : array-like
            The horizontal u-vector in the detector frame. Must be a unit
            vector."""
        if not self.xp.allclose(np.linalg.norm(value), 1.0):
            raise ValueError("`u` must be a unit vector. Consider normalizing"
                             " `u` using `u /= np.linalg.norm(u)`.")
        if self.xp.allclose(value, 0.0):
            raise ValueError("`u` must not be zero.")
        # TODO(Adriaan): can only be checked when u, v are set at the same
        #  time
        # if hasattr(self, '_v'):
        #     if not self.xp.allclose(self.xp.dot(value, self.v), 0.):
        #         raise ValueError("`u` and `v` must be orthogonal.")
        self._u = value

    @property
    def v(self):
        """Vertical v-vector in the detector frame of unit length."""
        return self._v

    @v.setter
    def v(self, value):
        """Set the vertical v-vector in the detector frame.

        Parameters
        ----------
        value : array-like
            The vertical v-vector in the detector frame. Must be a unit
            vector."""
        if not self.xp.allclose(np.linalg.norm(value), 1.0):
            raise ValueError("`v` must be a unit vector. Consider normalizing"
                             " `v` using `v /= np.linalg.norm(v)`.")
        if self.xp.allclose(value, 0.0):
            raise ValueError("`v` must not be zero.")
        # TODO(Adriaan): can only be checked when u, v are set at the same
        #  time
        # if hasattr(self, '_u'):
        #     if not self.xp.allclose(self.xp.dot(value, self.u), 0.):
        #         raise ValueError("`u` and `v` must be orthogonal.")
        self._v = value

    def __deepcopy__(self, memodict={}):
        obj = self.__class__(
            source_position=self.xp.copy(self.source_position),
            detector_position=self.xp.copy(self.detector_position),
            u=self.xp.copy(self.u),
            v=self.xp.copy(self.v),
            detector=Detector(
                rows=self.detector.rows,
                cols=self.detector.cols,
                pixel_width=self.detector.pixel_width,
                pixel_height=self.detector.pixel_height,
            ),
            array_module=self.xp,
        )
        return obj

    def __str__(self):
        return (
            f"Source {self.source_position} "
            f"Detector {self.detector_position} "
            f"Horizontal vector: {self.u} "
            f"Vertical vector: {self.v} "
        )


@dataclass
class ProjectionGeometrySequence:
    """Structure-of-arrays geometry data object.

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
    beam : array-like
        The beam type.
    xp : array-like, optional
        The array library to use. If None, it defaults to numpy.
    """

    xp = np

    @dataclass
    class DetectorSequence:
        """Structure-of-arrays detector data object.

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
    beam: Beam

    def __len__(self):
        return len(self.source_position)

    @property
    def detector_extent_min(self):
        """The minimum extent of the detector in the detector frame.
        This is the bottom left corner of the detector."""
        return (
            self.detector_position
            - self.v * self.detector.height[..., self.xp.newaxis] / 2
            - self.u * self.detector.width[..., self.xp.newaxis] / 2
        )

    @classmethod
    def fromList(cls, geometries: list[ProjectionGeometry], xp=None):
        """Create a ProjectionGeometrySequence from a list of ProjectionGeometry objects.

        Parameters
        ----------
        geometries : list[ProjectionGeometry]
            A list of ProjectionGeometry objects.
        xp : array-like, optional
            The array library to use. If None, it defaults to numpy.
        """
        if xp is None:
            xp = cls.xp

        assert np.all([geometries[0].beam == g.beam for g in geometries])

        def _cvrt(arr, dtype):
            return xp.ascontiguousarray(xp.array(arr, dtype=dtype))

        ds = cls.DetectorSequence(
            rows=_cvrt([g.detector.rows for g in geometries], xp.int32),
            cols=_cvrt([g.detector.cols for g in geometries], xp.int32),
            pixel_width=_cvrt([g.detector.pixel_width for g in geometries], xp.float32),
            pixel_height=_cvrt(
                [g.detector.pixel_height for g in geometries], xp.float32
            ),
        )
        gs = cls(
            source_position=_cvrt([g.source_position for g in geometries], xp.float32),
            detector_position=_cvrt(
                [g.detector_position for g in geometries], xp.float32
            ),
            u=_cvrt([g.u for g in geometries], xp.float32),
            v=_cvrt([g.v for g in geometries], xp.float32),
            detector=ds,
            beam=geometries[0].beam,
        )
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
            pixel_height=self.detector.pixel_height[indices],
        )
        gs = ProjectionGeometrySequence(
            source_position=self.source_position[indices],
            detector_position=self.detector_position[indices],
            u=self.u[indices],
            v=self.v[indices],
            detector=ds,
            beam=self.beam
        )
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
            beam=xp.copy(self.beam),
            detector=ProjectionGeometrySequence.DetectorSequence(
                rows=xp.copy(self.detector.rows),
                cols=xp.copy(self.detector.cols),
                pixel_width=xp.copy(self.detector.pixel_width),
                pixel_height=xp.copy(self.detector.pixel_height),
            ),
        )
        obj.xp = xp
        obj.detector.xp = xp
        return obj


def shift(geom: ProjectionGeometry, vector: np.ndarray) -> ProjectionGeometry:
    """Creates a new geometry by a 3D shift.

    Parameters
    ----------
    geom: ProjectionGeometry or ProjectionGeometrySequence
    vector: numpy.ndarray, cupy.ndarray, shape (3,), or (N, 3)
    for GeometrySequence

    Returns
    -------
    New `ProjectionGeometry` or `GeometrySequence`
    """
    geom = copy.deepcopy(geom)
    geom.source_position += vector
    geom.detector_position += vector
    return geom


def shift_(geom: ProjectionGeometry, vector: np.ndarray) -> None:
    """In-place shift of a geometry or geometry sequence.

    Parameters
    ----------
    geom: ProjectionGeometry or ProjectionGeometrySequence
    vector: numpy.ndarray, cupy.ndarray, shape (3,), or (N, 3)
    for GeometrySequence
    """
    geom.source_position += vector
    geom.detector_position += vector


def scale(geom: ProjectionGeometry, factor: np.ndarray) -> ProjectionGeometry:
    """Creates a new projection geometry by factor.

    Parameters
    ----------
    geom : ProjectionGeometry
        Projection geometry or geometry sequence.
    factor : float
        Factor by which to enlarge or shrink the geometry.

    Returns
    -------
    New `ProjectionGeometry`
    """
    geom = copy.deepcopy(geom)

    # detector pixels have to be scaled first, because
    # detector.width and detector.height need to be scaled accordingly
    horiz_pixel_vector = (geom.u * geom.detector.pixel_width) / factor
    new_pixel_width = np.linalg.norm(horiz_pixel_vector)
    new_u_unit = horiz_pixel_vector / new_pixel_width
    geom.detector.pixel_width = new_pixel_width
    geom.u = new_u_unit

    vert_pixel_vector = (geom.v * geom.detector.pixel_height) / factor
    new_pixel_height = np.linalg.norm(vert_pixel_vector)
    new_v_unit = vert_pixel_vector / new_pixel_height
    geom.detector.pixel_height = new_pixel_height
    geom.v = new_v_unit

    geom.source_position[:] = geom.source_position[:] / factor
    geom.detector_position[:] = geom.detector_position[:] / factor
    return geom


def scale_(geom: ProjectionGeometry, factor: float) -> None:
    """In-place factor of geometry or geometry sequence.

    Parameters
    ----------
    geom : ProjectionGeometry or ProjectionGeometrySequence
    factor : float
        Factor by which to enlarge or shrink the geometry
    """
    # detector pixels have to be scaled first, because
    # detector.width and detector.height need to be scaled accordingly
    xp = geom.xp

    pixel_vec = geom.u * geom.detector.pixel_width[..., xp.newaxis]
    pixel_vec /= factor

    geom.detector.pixel_width = xp.linalg.norm(pixel_vec, axis=-1)
    xp.divide(pixel_vec, geom.detector.pixel_width[..., xp.newaxis], out=geom.u)
    xp.multiply(geom.v, geom.detector.pixel_height[..., xp.newaxis], out=pixel_vec)
    pixel_vec /= factor

    geom.detector.pixel_height = xp.linalg.norm(pixel_vec, axis=-1)
    xp.divide(pixel_vec, geom.detector.pixel_height[..., xp.newaxis], out=geom.v)
    xp.divide(geom.source_position, factor, out=geom.source_position)
    xp.divide(geom.detector_position, factor, out=geom.detector_position)


def rotate(
    geom: ProjectionGeometry, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0
) -> ProjectionGeometry:
    """Creates a new geometry by rotation.

    Parameters
    ----------
    geom : ProjectionGeometry or ProjectionGeometrySequence
    roll : float
        Rotation around x-axis
    pitch : float
        Rotation around y-axis
    yaw : float
        Rotation around z-axis

    Returns
    -------
    New `ProjectionGeometry`
    """
    ngeom = copy.deepcopy(geom)

    R = angles2mat(roll, pitch, yaw)
    ngeom.source_position = R @ geom.source_position
    ngeom.detector_position = R @ geom.detector_position
    ngeom.u = R @ geom.u
    ngeom.v = R @ geom.v
    ngeom.beam = geom.beam
    return ngeom


def rotate_(
    geom: ProjectionGeometry, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0
) -> None:
    """In-place rotation of a geometry.

    Parameters
    ----------
    geom : ProjectionGeometry or ProjectionGeometrySequence
    roll : float
        Rotation around x-axis
    pitch : float
        Rotation around y-axis
    yaw : float
        Rotation around z-axis
    """
    R = geom.xp.asarray(angles2mat(roll, pitch, yaw), dtype=geom.source_position.dtype)
    geom.source_position[...] = geom.source_position @ R
    geom.detector_position[...] = geom.detector_position @ R
    geom.u[...] = geom.u @ R
    geom.v[...] = geom.v @ R
