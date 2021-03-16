"""
This file originates from ReFleX.
This should probably become a separate package.
"""

from abc import ABC

import numpy as np
import transforms3d


class Flat1DDetector:
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

    @pixel_width.setter
    def pixel_width(self, value):
        self._pixel_width = value

    @property
    def nr_pixels(self):
        return self._nr_pixels

    @property
    def width(self):
        return self.pixel_width * self.nr_pixels


class Static2DGeometry:
    ANGLES_CONVENTION = "sxyz"

    def __init__(self,
                 tube_pos: np.ndarray,
                 det_pos: np.ndarray,
                 det_rotation: float,
                 detector: Flat1DDetector):
        self.__det_pos = det_pos
        self.__tube_pos = tube_pos
        self.__detector = detector
        self.__det_rotation = det_rotation

    @property
    def tube_position(self) -> np.ndarray:
        return self.__tube_pos

    @property
    def detector_position(self) -> np.ndarray:
        return self.__det_pos

    @property
    def detector(self) -> Flat1DDetector:
        return self.__detector

    @property
    def detector_rotation(self) -> float:
        return self.__det_rotation

    @property
    def stage_position(self):
        return np.zeros_like(self.__tube_pos)

    @property
    def stage_rotation(self) -> float:
        return 0.

    @staticmethod
    def angles2mat(r, p, y) -> np.ndarray:
        return transforms3d.euler.euler2mat(
            r, p, y,
            Static2DGeometry.ANGLES_CONVENTION
        )

    @staticmethod
    def mat2angles(mat) -> tuple:
        return transforms3d.euler.mat2euler(
            mat,
            Static2DGeometry.ANGLES_CONVENTION
        )

    @property
    def u(self):
        """Horizontal u-vector in the detector frame."""
        R = Static2DGeometry.angles2mat(0., 0., self.__det_rotation)
        return R[:2, :2] @ [0., 1.]

    # def _angle_to_matrix(self, rot) -> np.ndarray:
    #     R = transforms3d.axangles.axangle2mat([0, 0, self.DET_AXIS[0]], rot)
    #     return R[:2, :2].T


class Base2DDecorator(Static2DGeometry, ABC):
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

    def __init__(self, decorated_geometry: Static2DGeometry):
        self._g = decorated_geometry

    @property
    def tube_position(self) -> np.ndarray:
        return self._g.tube_position

    @property
    def detector_position(self) -> np.ndarray:
        return self._g.detector_position

    @property
    def detector(self) -> Flat1DDetector:
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


class rotate(Base2DDecorator):
    """Decorator for StaticGeometry that rotates around (0,0,0) rotated coordinate frame by specifying a rotation matrix."""

    def __init__(self, geom: Static2DGeometry, rotation_matrix: np.ndarray):
        """Rotation around (0,0,0).
        @todo Also support argument around=..., to rotate around different vectors
        """
        self._g = geom
        self._rotation_matrix = rotation_matrix

        Base2DDecorator.__init__(self, geom)

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


class shift(Base2DDecorator):
    """Decorates the StaticGeometry with a shift of all the positions in the system.
    It does not affect the rotations in the StaticGeometry, because they are all intrinsic."""

    def __init__(self, geom: Static2DGeometry, shift_vector: np.ndarray):
        Base2DDecorator.__init__(self, geom)
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


class rotate_stage(Base2DDecorator):
    """Decorates the StaticGeometry with a rotation around the real, tilted or translated axis of the rotation stage.
    it is a basic modification of the stage_rotation property. It does not modify the roll/pitch of the stage."""

    def __init__(self, previous_geometry: Static2DGeometry, added_angle=None):
        self._added_angle = added_angle

        Base2DDecorator.__init__(self, previous_geometry)

    @property
    def stage_rotation(self):
        return self._g.stage_rotation + self._added_angle


class centralize(Base2DDecorator):
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

        Base2DDecorator.__init__(self, geom)


def plot(geoms: dict):
    import matplotlib.pyplot as plt

    tubes_x = [g.tube_position[0] for g in geoms.values()]
    tubes_y = [g.tube_position[1] for g in geoms.values()]
    dets_x = [g.detector_position[0] for g in geoms.values()]
    dets_y = [g.detector_position[1] for g in geoms.values()]

    plt.figure()
    plt.scatter(tubes_x, tubes_y)
    plt.scatter(dets_x, dets_y)
    for g in geoms.values():
        plt.arrow(*g.tube_position, *g.u * g.detector.nr_pixels)

    plt.show()
