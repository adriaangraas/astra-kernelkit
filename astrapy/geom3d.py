import copy
from typing import Sequence

import numpy as np
import transforms3d


class Detector:
    """A baseclass for detectors without any special properties.
    Note that you can initialize detectors with or without binned pixels,
    rows and cols, there is no distinction on the software level.
    """

    def __init__(self, rows, cols, pixel_width, pixel_height):
        self.rows = rows
        self.cols = cols
        self.pixel_width = pixel_width
        self.pixel_height = pixel_height

    @property
    def width(self):
        return self.pixel_width * self.cols

    @property
    def height(self):
        return self.pixel_height * self.rows

    @property
    def pixel_volume(self):
        return self.pixel_width * self.pixel_height


class Static3DGeometry:
    ANGLES_CONVENTION = "sxyz"

    def __init__(self,
                 tube_pos: np.ndarray,
                 det_pos: np.ndarray,
                 det_roll: float,
                 det_pitch: float,
                 det_yaw: float,
                 detector: Detector,
                 det_piercing: Sequence = None):
        self.tube_position = tube_pos
        self.detector_position = det_pos
        self.detector_piercing = det_piercing if det_piercing else det_pos
        self.detector = detector
        self.__det_roll = det_roll
        self.__det_pitch = det_pitch
        self.__det_yaw = det_yaw

    @property
    def detector_extent_min(self):
        return (self.detector_position
                - self.v * self.detector.height / 2
                - self.u * self.detector.width / 2)

    @property
    def detector_roll(self):
        return self.__det_roll

    @property
    def detector_pitch(self):
        return self.__det_pitch

    @property
    def detector_yaw(self):
        return self.__det_yaw

    @property
    def u(self):
        """Horizontal u-vector in the detector frame."""
        R = Static3DGeometry.angles2mat(self.__det_roll,
                                        self.__det_pitch,
                                        self.__det_yaw)
        return R @ [0, 1, 0]

    @property
    def v(self):
        """Vertical v-vector in the detector frame."""
        R = Static3DGeometry.angles2mat(self.__det_roll,
                                        self.__det_pitch,
                                        self.__det_yaw)
        return R @ [0, 0, 1]

    @staticmethod
    def angles2mat(r, p, y) -> np.ndarray:
        return transforms3d.euler.euler2mat(
            r, p, y,
            Static3DGeometry.ANGLES_CONVENTION
        )

    @staticmethod
    def mat2angles(mat) -> tuple:
        return transforms3d.euler.mat2euler(
            mat,
            Static3DGeometry.ANGLES_CONVENTION
        )

    def __str__(self):
        return f"Tube {self.tube_position} " + \
               f"Detector {self.detector_position}"


class Geometry(Static3DGeometry):
    def __init__(self,
                 tube_pos: Sequence,
                 det_pos: Sequence,
                 u_unit: Sequence,
                 v_unit: Sequence,
                 detector,
                 det_piercing=None):
        # TODO(Adriaan): unify __init__ or the classes
        self.tube_position = np.array(tube_pos)
        self.detector_position = np.array(det_pos)
        # TODO(Adriaan): detector piercing is only useful for cone?
        self.detector_piercing = det_piercing if det_piercing else det_pos
        self.detector = detector
        self.u = np.array(u_unit)
        self.v = np.array(v_unit)

    @property
    def u(self):
        return self.__u

    @u.setter
    def u(self, value):
        if not np.allclose(np.linalg.norm(value), 1.):
            raise ValueError("`u` must be a unit vector.")

        self.__u = value

    @property
    def v(self):
        return self.__v

    @v.setter
    def v(self, value):
        if not np.allclose(np.linalg.norm(value), 1.):
            raise ValueError("`v` must be a unit vector.")

        self.__v = value

    def __str__(self):
        return f"Tube {self.tube_position} " + \
               f"Detector {self.detector_position}" + \
               f"U {self.u}" + \
               f"V {self.v}"


def shift(geom: Static3DGeometry, shift_vector: np.ndarray):
    geom.tube_position += shift_vector
    geom.detector_position += shift_vector


def scale(geom: Static3DGeometry, scaling: np.ndarray):
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

    geom.tube_position[:] = geom.tube_position[:] / scaling
    geom.detector_position[:] = geom.detector_position[:] / scaling


def rotate_inplace(geom: Static3DGeometry,
                   roll: float = 0., pitch: float = 0., yaw: float = 0.):
    RT = Static3DGeometry.angles2mat(roll, pitch, yaw).T
    geom.tube_position = RT @ geom.tube_position
    geom.detector_position = RT @ geom.detector_position
    geom.u = RT @ geom.u
    geom.v = RT @ geom.v


def rotate(geom: Static3DGeometry,
           roll: float = 0., pitch: float = 0., yaw: float = 0.):
    geom = copy.deepcopy(geom)
    RT = Static3DGeometry.angles2mat(roll, pitch, yaw).T
    geom.tube_position = RT @ geom.tube_position
    geom.detector_position = RT @ geom.detector_position
    geom.u = RT @ geom.u
    geom.v = RT @ geom.v
    return geom


def plot(geoms: dict):
    import matplotlib.pyplot as plt

    tubes_x = [g.tube_position[0] for g in geoms]
    tubes_y = [g.tube_position[1] for g in geoms]
    tubes_z = [g.tube_position[2] for g in geoms]
    dets_x = [g.detector_position[0] for g in geoms]
    dets_y = [g.detector_position[1] for g in geoms]
    dets_z = [g.detector_position[2] for g in geoms]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter(tubes_x, tubes_y, tubes_z, marker='*')
    ax.scatter(dets_x, dets_y, dets_z, marker='o')
    # for g in geoms:
    #     ax.arrow3D(*g.tube_position, *g.u * g.detector.width)
    #     ax.arrow3D(*g.tube_position, *g.v * g.detector.height)

    plt.show()
