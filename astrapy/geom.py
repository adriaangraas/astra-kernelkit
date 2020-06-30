"""
This file originates from ReFleX.
This should probably become a separate package.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
import transforms3d


class FlatDetector(ABC):
    # Reference axis of the detector plane w.r.t. the world frame
    # see: BaseStaticGeometry
    AXIS_HORIZONTAL = [0, 1, 0]
    AXIS_VERTICAL = [0, 0, 1]

    @abstractmethod
    def pixel_width(self):
        pass


class Flat1DDetector(FlatDetector):
    def __init__(self, nr_pixels, pixel_width):
        """
        :param rows: Number of horizontal rows
        :param cols: Number of vertical columns
        :param pixel_size: Total pixel size
        """
        self._nr_pixels = nr_pixels
        self._pixel_width = pixel_width

    @property
    def pixel_width(self):
        return self._pixel_width

    @property
    def nr_pixels(self):
        return self._nr_pixels

    @property
    def width(self):
        return self.pixel_width * self.nr_pixels


class Flat2DDetector(FlatDetector):
    """A baseclass for detectors without any special properties.
    Note that you can initialize detectors with or without binned pixels, rows and cols, there is no distinction on
    the software level.
    """

    def __init__(self, rows, cols, pixel_width, pixel_height):
        """
        :param rows: Number of horizontal rows
        :param cols: Number of vertical columns
        :param pixel_size: Total pixel size
        """
        self._rows = rows
        self._cols = cols
        self._pixel_width = pixel_width
        self._pixel_height = pixel_height

    @property
    def pixel_width(self):
        return self._pixel_width

    @property
    def pixel_height(self):
        return self._pixel_height

    @property
    def rows(self):
        return self._rows

    @property
    def cols(self):
        return self._cols

    @property
    def width(self):
        return self.pixel_width * self.cols

    @property
    def height(self):
        return self.pixel_height * self.rows


class SquarePixelDetector(Flat2DDetector):
    """A flat detector of which pixel height and width are equal."""

    def __init__(self, rows, cols, pixel_size):
        """
        :param rows: Number of horizontal rows
        :param cols: Number of vertical columns
        :param pixel_size: Total pixel size
        """
        self._rows = rows
        self._cols = cols
        self._pixel_size = pixel_size

    @property
    def pixel_width(self):
        return self._pixel_size

    @property
    def pixel_height(self):
        return self._pixel_size


class Rotatable3DGeometryMixin:
    """A Mixin for Geometry classes to compute rotations.

    This mixin is required to share functionality between the StaticGeometry class and its decorators."""

    DET_ROLL_AXIS = [-1, 0, 0]  # the face of the detector is in the negative x-axis
    DET_PITCH_AXIS = [0, 1, 0]
    DET_YAW_AXIS = [0, 0, 1]
    STAGE_ROLL_AXIS = [1, 0, 0]
    STAGE_PITCH_AXIS = [0, 1, 0]
    STAGE_YAW_AXIS = [0, 0, 1]

    @property
    def stage_rotation_matrix(self) -> np.ndarray:
        return angles_to_matrix(roll=self.stage_roll,
                                roll_axis=self.STAGE_ROLL_AXIS,
                                pitch=self.stage_pitch,
                                pitch_axis=self.STAGE_PITCH_AXIS,
                                yaw=self.stage_rotation,
                                yaw_axis=self.STAGE_YAW_AXIS)

    @property
    def detector_rotation_matrix(self) -> np.ndarray:
        return angles_to_matrix(roll=self.detector_roll,
                                roll_axis=self.DET_ROLL_AXIS,
                                pitch=self.detector_pitch,
                                pitch_axis=self.DET_PITCH_AXIS,
                                yaw=self.detector_yaw,
                                yaw_axis=self.DET_YAW_AXIS)


class Rotatable2DGeometryMixin:
    """A Mixin for Geometry classes to compute rotations.

    This mixin is required to share functionality between the StaticGeometry class and its decorators."""

    DET_ROLL_AXIS = [-1, 0, 0]  # the face of the detector is in the negative x-axis
    DET_PITCH_AXIS = [0, 1, 0]
    DET_YAW_AXIS = [0, 0, 1]
    STAGE_ROLL_AXIS = [1, 0, 0]
    STAGE_PITCH_AXIS = [0, 1, 0]
    STAGE_YAW_AXIS = [0, 0, 1]


    @property
    def detector_rotation_matrix(self) -> np.ndarray:
        return angle_to_matrix(self.detector_rotation)

    def _angle_to_matrix(self, rot) -> np.ndarray:
        roll_matrix = transforms3d.axangles.axangle2mat(roll_axis, roll)
        pitch_axis = np.dot(roll_matrix, pitch_axis)
        roll_pitch_matrix = transforms3d.axangles.axangle2mat(pitch_axis, pitch)
        yaw_axis = np.dot(roll_pitch_matrix, yaw_axis)
        roll_pitch_yaw_matrix = transforms3d.axangles.axangle2mat(yaw_axis, yaw)

        return roll_pitch_yaw_matrix.t


class NdGeometry(ABC):
    """A static set-up of the FleX-ray scanner.
    That is: one fully flexible tube, detector, and rotation stage.

    This is essentially a bunch of variables that encodes all positions, and also incorporates
    information for a tilted stage or detector.

    It also standardizes, x, y, z directions and rotation conventions.
    x: magnification axis (source-detector line) (positive towards detector)
    y: horizontal axis, orthogonal to magnification (positive in depth)
    z: vertical axis, orthogonal to magnification (positive upwards)
    """

    WORLD_X_AXIS = [1, 0, 0]
    WORLD_Y_AXIS = [0, 1, 0]
    WORLD_Z_AXIS = [0, 0, 1]

    @property
    @abstractmethod
    def tube_position(self):
        pass

    @property
    @abstractmethod
    def detector_position(self):
        pass

    @property
    @abstractmethod
    def stage_position(self):
        pass

    @property
    @abstractmethod
    def stage_roll(self):
        """Intrinsic to the stage world frame."""
        pass

    @property
    @abstractmethod
    def stage_pitch(self):
        """Intrinsic to the stage frame, but after roll."""
        pass

    @property
    @abstractmethod
    def stage_rotation(self):
        """Intrinsic to the stage frame.
        Note this rotation is described around the tilted (roll-pitched) axis, and not w.r.t. world coordinates."""
        pass

    @property
    @abstractmethod
    def detector_roll(self):
        """Intrinsic to the detector frame."""
        pass

    @property
    @abstractmethod
    def detector_pitch(self):
        """Intrinsic to the detector frame, after roll."""
        pass

    @property
    @abstractmethod
    def detector_yaw(self):
        """Intrinsic to the detector frame, after pitch."""
        pass

    @property
    @abstractmethod
    def detector(self) -> FlatDetector:
        pass

    def __str__(self):
        return "Tube: X,Y,Z:" + str(self.tube_position) + "\n" + \
               "Stage: X,Y,Z:" + str(self.stage_position) + f" R,P:{self.stage_roll, self.stage_pitch}\n" + \
               f"Detector {type(self.detector)}: X,Y,Z:" + str(
            self.detector_position) + f" R,P,Y:{self.detector_roll},{self.detector_pitch},{self.detector_yaw}\n"


class Static2DGeometry(NdGeometry, Rotatable2DGeometryMixin):
    def __init__(self, det_pos: np.ndarray, tube_pos: np.ndarray, det_yaw: float, detector: FlatDetector):
        self.__det_pos = det_pos
        self.__tube_pos = tube_pos
        self.__detector = detector
        self.__det_yaw = det_yaw

    @property
    def tube_position(self) -> np.ndarray:
        return self.__tube_pos

    @property
    def detector_position(self) -> np.ndarray:
        return self.__det_pos

    @detector_position.setter
    def detector_position(self, value: np.ndarray):
        self.__det_pos = value

    @property
    def detector(self) -> FlatDetector:
        return self.__detector

    # @todo is there a nicer way?
    # def to_list(self):
    #     det_u_x = self.detector_rotation_matrix
    #     det_u_y =
    #     return [*self.__det_pos, self.det_u_x, self.det_u_y, *self.__tube_pos]

    @property
    def detector_pitch(self):
        return 0  # in 2D

    @property
    def detector_roll(self):
        return 0  # in 2D

    @property
    def detector_yaw(self):
        return self.__det_yaw

    @property
    def stage_pitch(self):
        raise NotImplementedError()

    @property
    def stage_roll(self):
        raise NotImplementedError()

    @property
    def stage_position(self):
        return np.zeros_like(self.__tube_pos)

    @property
    def stage_rotation(self):
        return 0

    # def __repr__(self):
    #     return f"Det={self.__det_x};{self.__det_y};u={self.det_u_x};{self.det_u_y} Tube={self.__tube_x};{self.__tube_y}"


class GeometryDict:
    """
    Dictionary of FleXray geometries for every time or angle.

    This is multi-interpretable. The dictionary keys can be
     - an ordering of the dynamics
     - time
     - the base for multiple detectors or sources

    I assume for now that during time the detector cannot switch capturing mode, or ROI. If it happens, this class
    should be extended."""

    def __init__(self):
        self._geoms_at_times = dict()

    @property
    def lasttime(self):
        # Getting the last time known to the dynamic geometry .
        # Example: dyngeom[geom.lasttime].stage_rotation() for example.
        return max(self._geoms_at_times.keys())

    def __getitem__(self, t: float):
        return self._geoms_at_times[t]

    def __setitem__(self, t: float, geom: Static2DGeometry):
        self._geoms_at_times[t] = geom

    def __delitem__(self, t):
        del self._geoms_at_times[t]

    def __len__(self):
        return len(self._geoms_at_times)

    def to_dict(self):
        return OrderedDict(sorted(self._geoms_at_times.items()))

    def _static_geometry(self, t: float = 0.0):
        return self._geoms_at_times[t]

    def tube_position(self, t: float = 0.0):
        return self._static_geometry(t).tube_position

    def detector_position(self, t: float = 0.0):
        return self._static_geometry(t).detector_position

    def stage_position(self, t: float = 0.0):
        return self._static_geometry(t).stage_position

    def detector(self, t: float = 0.0):
        return self._static_geometry(t).detector

    def detector_roll(self, t: float = 0.0):
        return self._static_geometry(t).detector_roll

    def detector_yaw(self, t: float = 0.0):
        return self._static_geometry(t).detector_yaw

    def detector_pitch(self, t: float = 0.0):
        return self._static_geometry(t).detector_pitch

    def detector_rotation_matrix(self, t: float = 0.0):
        return self._static_geometry(t).detector_rotation_matrix

    def stage_rotation_matrix(self, t: float = 0.0):
        return self._static_geometry(t).stage_rotation_matrix


class BaseDecorator(NdGeometry, ABC):
    """This is an "abstract base decorator". It says: every class that wants to provide
    adjustments for StaticGeometry will automatically have methods that are in the StaticGeometry.

    Why do we need this? If the geometry changes dynamically, because the object rotates, detector
    moves or something else, we can just of course copy the StaticGeometry object and put new values
    there. But then we lost "what we did", which could have been just an elementary operation,
    like moving the detector a bit. This is useful information that we might need later.

    A decorator is a run-time addition on an object. In a decorator we describe what we changed
    in the original object. So, here is an simplified example (will not work):
        geometry = StaticGeometry(source=[0,0,0], detector=[10,10,10])
        moved_geometry = MoveDetectorDecorator(geometry, [0, 0, 10])
        print(moved_geometry.detector_position)  # [10,10,20]
    Now moved_geometry is our changed geometry, that just contains what we changed,
    namely moving the detector by [0,0,10].

    This class, the `BaseDecorator` helps writing any decorator, the shared implementation of what every
    decorator should have, namely returning the properties of the object that it is decorating.
    """

    def __init__(self, decorated_geometry: NdGeometry):
        self._g = decorated_geometry

    @property
    def tube_position(self) -> np.ndarray:
        return self._g.tube_position

    @property
    def detector_position(self) -> np.ndarray:
        return self._g.detector_position

    @property
    def detector(self) -> FlatDetector:
        return self._g.detector

    @property
    def detector_pitch(self) -> float:
        return self._g.detector_pitch

    @property
    def detector_roll(self) -> float:
        return self._g.detector_roll

    @property
    def detector_yaw(self) -> float:
        return self._g.detector_yaw

    @property
    def stage_pitch(self) -> float:
        return self._g.stage_pitch

    @property
    def stage_position(self) -> np.ndarray:
        return self._g.stage_position

    @property
    def stage_roll(self) -> float:
        return self._g.stage_roll

    @property
    def stage_rotation(self) -> float:
        return self._g.stage_rotation


class rotate(BaseDecorator):
    """Decorator for StaticGeometry that rotates around (0,0,0) rotated coordinate frame by specifying a rotation matrix."""

    def __init__(self, geom: Static2DGeometry, rotation_matrix: np.ndarray):
        """Rotation around (0,0,0).
        @todo Also support argument around=..., to rotate around different vectors
        """
        self._g = geom
        self._rotation_matrix = rotation_matrix

        BaseDecorator.__init__(self, geom)

    def _rotate(self, vector):
        """Internal helper to perform the rotations"""
        return np.dot(self._rotation_matrix, vector)

    @property
    def tube_position(self) -> np.ndarray:
        return self._rotate(self._g.tube_position)

    @property
    def detector_position(self) -> np.ndarray:
        return self._rotate(self._g.detector_position)

    @property
    def stage_position(self) -> np.ndarray:
        return self._rotate(self._g.stage_position)

    @property
    def detector_rotation_matrix(self) -> np.ndarray:
        return self._rotation_matrix @ self._g.detector_rotation_matrix

    @property
    def stage_rotation_matrix(self) -> np.ndarray:
        return self._rotation_matrix @ self._g.stage_rotation_matrix


class shift(BaseDecorator):
    """Decorates the StaticGeometry with a shift of all the positions in the system.
    It does not affect the rotations in the StaticGeometry, because they are all intrinsic."""

    def __init__(self, geom: Static2DGeometry, shift_vector: np.ndarray):
        BaseDecorator.__init__(self, geom)
        self._shift = shift_vector

    @property
    def tube_position(self) -> np.ndarray:
        return self._g.tube_position + self._shift

    @property
    def stage_position(self) -> np.ndarray:
        return self._g.stage_position + self._shift

    @property
    def detector_position(self) -> np.ndarray:
        return self._g.detector_position + self._shift


class rotate_stage(BaseDecorator):
    """Decorates the StaticGeometry with a rotation around the real, tilted or translated axis of the rotation stage.
    it is a basic modification of the stage_rotation property. It does not modify the roll/pitch of the stage."""

    def __init__(self, previous_geometry: Static2DGeometry, added_angle=None):
        self._added_angle = added_angle

        BaseDecorator.__init__(self, previous_geometry)

    @property
    def stage_rotation(self):
        return self._g.stage_rotation + self._added_angle


class centralize(BaseDecorator):
    """
    Vectors in this geometry are not with respect to the "world coordinate system"
    but with respect to the intrinsic coordinate system at the object rotation stage.

    This might be a bit confusing, because source and detector are moving around the stage while the stage is
    moving (and wobbling) itself.

    For instance, if the stage is moving a bit this is reflected in the source and detector
    taking different distances to the stage.
    """

    def __init__(self, geom: Static2DGeometry):
        # first we shift so that the object is central
        geom = shift(geom, -np.array(geom.stage_position))

        # then lower the all positions so that the tube is in the horizontal plane
        # @todo it looks like ASTRA hangs itself when this line is not there, is this a bug in ASTRA?
        geom = shift(geom, -np.array([0, 0, geom.tube_position[2]]))

        # then rotate around (0,0,0) according to roll-pitch-yaw conventions
        geom = rotate(geom, geom.stage_rotation_matrix)

        BaseDecorator.__init__(self, geom)


def dynamic_geom_to_3d_astra_vectors(geom: GeometryDict) -> np.ndarray:
    """This method converts to  FleXrayGeometry in a 12-component parametrization
    Should be useful if you want to convert to the ASTRA vector `vec_cone` geometry.
    """
    vs = np.empty((len(geom), 12))

    for i, (t, static_geom) in enumerate(geom.to_dict().items()):
        if not isinstance(geom.detector(0), Flat2DDetector):
            raise ValueError("Only works for 3D detectors.")
            # todo: I guess this can be done handled in a nicer way

        g = centralize(static_geom)
        vs[i, 0:3] = [g.tube_position[1], g.tube_position[0], g.tube_position[2]]
        vs[i, 3:6] = [g.detector_position[1], g.detector_position[0], g.detector_position[2]]

        horizontal_pixel_vector = np.multiply(g.detector.pixel_width, FlatDetector.AXIS_HORIZONTAL)
        hv = g.stage_rotation_matrix @ horizontal_pixel_vector
        vs[i, 6:9] = [hv[1], hv[0], hv[2]]

        vertical_pixel_vector = np.multiply(g.detector.pixel_width, FlatDetector.AXIS_VERTICAL)
        vv = g.stage_rotation_matrix @ vertical_pixel_vector
        vs[i, 9:12] = [vv[1], vv[0], vv[2]]

    return vs


def angles_to_matrix(roll=0.0, roll_axis=NdGeometry.WORLD_X_AXIS,
                     pitch=0.0, pitch_axis=NdGeometry.WORLD_Y_AXIS,
                     yaw=0.0, yaw_axis=NdGeometry.WORLD_Z_AXIS) -> np.ndarray:
    """returns matrix that takes in a vector in the skewed space and outputs
    a vector in the corrected space.
    this function makes sure that the convention
    roll-pitch-yaw, rpy, in this order, is always maintained.
    intrinsic rotations (see https://en.wikipedia.org/wiki/euler_angles)"""
    roll_matrix = transforms3d.axangles.axangle2mat(roll_axis, roll)
    pitch_axis = np.dot(roll_matrix, pitch_axis)
    roll_pitch_matrix = transforms3d.axangles.axangle2mat(pitch_axis, pitch)
    yaw_axis = np.dot(roll_pitch_matrix, yaw_axis)
    roll_pitch_yaw_matrix = transforms3d.axangles.axangle2mat(yaw_axis, yaw)

    return roll_pitch_yaw_matrix.T

