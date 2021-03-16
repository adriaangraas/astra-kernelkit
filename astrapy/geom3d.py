from abc import ABC, abstractmethod
import numpy as np
import transforms3d


class Flat2DDetector:
    """A baseclass for detectors without any special properties.
    Note that you can initialize detectors with or without binned pixels,
    rows and cols, there is no distinction on the software level.
    """
    # Reference axis of the detector plane w.r.t. the world frame
    # see: BaseStaticGeometry
    AXIS_HORIZONTAL = [0, 1, 0]
    AXIS_VERTICAL = [0, 0, 1]


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

    @pixel_width.setter
    def pixel_width(self, value):
        self._pixel_width = value

    @property
    def pixel_height(self):
        return self._pixel_height

    @pixel_height.setter
    def pixel_height(self, value):
        self._pixel_height = value

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

    def __init__(self, rows, cols, pixel_size, pixel_width, pixel_height):
        """
        :param rows: Number of horizontal rows
        :param cols: Number of vertical columns
        :param pixel_size: Total pixel size
        """
        super().__init__(rows, cols, pixel_width, pixel_height)
        self._rows = rows
        self._cols = cols
        self._pixel_size = pixel_size

    @property
    def pixel_width(self):
        return self._pixel_size

    @property
    def pixel_height(self):
        return self._pixel_size


class Static3DGeometry:
    ANGLES_CONVENTION = "sxyz"

    def __init__(self,
                 tube_pos: np.ndarray,
                 det_pos: np.ndarray,
                 det_roll: float,
                 det_pitch: float,
                 det_yaw: float,
                 detector: Flat2DDetector):
        self.__det_pos = det_pos
        self.__tube_pos = tube_pos
        self.__detector = detector
        self.__det_roll = det_roll
        self.__det_pitch = det_pitch
        self.__det_yaw = det_yaw


    @property
    def tube_position(self):
        return self.__tube_pos

    @property
    def detector_position(self):
        return self.__det_pos

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
    def detector(self) -> Flat2DDetector:
        return self.__detector

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


class AstraStatic3DGeometry(Static3DGeometry):
    def __init__(self, tube_pos, det_pos,
                 u_unit,
                 v_unit,
                 detector):
        self.__tube_pos = tube_pos
        self.__det_pos = det_pos
        self.u = u_unit
        self.v = v_unit
        self.__det = detector

    @property
    def tube_position(self):
        return self.__tube_pos

    @property
    def detector_position(self):
        return self.__det_pos

    @property
    def detector_roll(self):
        raise NotImplementedError()

    @property
    def detector_pitch(self):
        raise NotImplementedError()

    @property
    def detector_yaw(self):
        raise NotImplementedError()

    @property
    def detector(self) -> Flat2DDetector:
        return self.__det

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
