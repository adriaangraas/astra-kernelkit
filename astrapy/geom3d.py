import numpy as np
import transforms3d


class Flat2DDetector:
    """A baseclass for detectors without any special properties.
    Note that you can initialize detectors with or without binned pixels,
    rows and cols, there is no distinction on the software level.
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

    @rows.setter
    def rows(self, value):
        self._rows = value

    @property
    def cols(self):
        return self._cols

    @cols.setter
    def cols(self, value):
        self._cols = value

    @property
    def width(self):
        return self.pixel_width * self.cols

    @property
    def height(self):
        return self.pixel_height * self.rows

    @property
    def pixel_volume(self):
        return self.pixel_width * self.pixel_height


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
    def detector_corner(self):
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
    def detector(self) -> Flat2DDetector:
        return self.__detector

    def main_axis(self) -> int:
        """Helper to determine the main ray-projection direction in
        conebeam geometries."""
        d = np.abs(self.tube_position - self.detector_position)
        if d[0] >= d[1] and d[0] >= d[2]:
            ax = 0
        elif d[1] >= d[0] and d[1] >= d[2]:
            ax = 1
        else:
            ax = 2

        return ax

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


def shift(geom: Static3DGeometry, shift_vector: np.ndarray):
    geom.tube_position[:] = geom.tube_position + shift_vector
    geom.detector_position[:] = geom.detector_position + shift_vector


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
