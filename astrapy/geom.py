import copy
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cupy as cp
import numpy as np
import transforms3d


@dataclass
class Detector:
    """A baseclass for detectors without any special properties.
    Note that you can initialize detectors with or without binned pixels,
    rows and cols, there is no distinction on the software level.
    """
    rows: int
    cols: int
    pixel_width: float
    pixel_height: float

    @property
    def width(self):
        return self.pixel_width * self.cols

    @property
    def height(self):
        return self.pixel_height * self.rows

    @property
    def pixel_volume(self):
        return self.pixel_width * self.pixel_height


class Geometry:
    ANGLES_CONVENTION = "sxyz"

    def __init__(self,
                 tube_pos: Sequence,
                 det_pos: Sequence,
                 u_unit: Sequence,
                 v_unit: Sequence,
                 detector,
                 det_piercing=None):
        self.tube_position = np.array(tube_pos)
        self.detector_position = np.array(det_pos)
        # TODO(Adriaan): detector piercing is only useful for cone?
        # self.detector_piercing = det_piercing if det_piercing else det_pos
        self.detector = detector
        self.u = np.array(u_unit)
        self.v = np.array(v_unit)

    @property
    def detector_extent_min(self):
        return (self.detector_position
                - self.v * self.detector.height / 2
                - self.u * self.detector.width / 2)

    # @property
    # def detector_roll(self):
    #     return self.__det_roll
    #
    # @property
    # def detector_pitch(self):
    #     return self.__det_pitch
    #
    # @property
    # def detector_yaw(self):
    #     return self.__det_yaw
    #
    # @property
    # def u(self):
    #     """Horizontal u-vector in the detector frame."""
    #     R = Static3DGeometry.angles2mat(self.__det_roll,
    #                                     self.__det_pitch,
    #                                     self.__det_yaw)
    #     return R @ [0, 1, 0]
    #
    # @property
    # def v(self):
    #     """Vertical v-vector in the detector frame."""
    #     R = Static3DGeometry.angles2mat(self.__det_roll,
    #                                     self.__det_pitch,
    #                                     self.__det_yaw)
    #     return R @ [0, 0, 1]


    @staticmethod
    def angles2mat(r, p, y) -> np.ndarray:
        return transforms3d.euler.euler2mat(
            r, p, y, Geometry.ANGLES_CONVENTION)

    @staticmethod
    def mat2angles(mat) -> float:
        return transforms3d.euler.mat2euler(
            mat, Geometry.ANGLES_CONVENTION)

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


@dataclass
class GeometrySequence:
    """Structure-of-arrays geometry data object"""
    xp = np

    @dataclass
    class DetectorSequence:
        xp = np
        rows: xp.ndarray
        cols: xp.ndarray
        pixel_width: xp.ndarray
        pixel_height: xp.ndarray

        @property
        def height(self):
            return self.pixel_height * self.rows

        @property
        def width(self):
            return self.pixel_width * self.cols

        @property
        def pixel_volume(self):
            return self.pixel_width * self.pixel_height

    tube_position: xp.ndarray
    detector_position: xp.ndarray
    u: xp.ndarray
    v: xp.ndarray
    detector: DetectorSequence

    def __len__(self):
        return len(self.tube_position)

    @property
    def detector_extent_min(self):
        return (self.detector_position
                - self.v * self.detector.height[..., self.xp.newaxis] / 2
                - self.u * self.detector.width[..., self.xp.newaxis] / 2)

    @classmethod
    def fromList(cls, geometries: List[Geometry]):
        _cvrt = lambda arr: cls.xp.ascontiguousarray(
            cls.xp.array(arr, dtype=cls.xp.float32))

        ds = cls.DetectorSequence(
            rows=_cvrt([g.detector.rows for g in geometries]),
            cols=_cvrt([g.detector.cols for g in geometries]),
            pixel_width=_cvrt([g.detector.pixel_width for g in geometries]),
            pixel_height=_cvrt([g.detector.pixel_height for g in geometries]))
        gs = cls(
            tube_position=_cvrt([g.tube_position for g in geometries]),
            detector_position=_cvrt([g.detector_position for g in geometries]),
            u=_cvrt([g.u for g in geometries]),
            v=_cvrt([g.v for g in geometries]),
            detector=ds)
        return gs

    def take(self, indices):
        ds = self.DetectorSequence(
            rows=self.detector.rows[indices],
            cols=self.detector.cols[indices],
            pixel_width=self.detector.pixel_width[indices],
            pixel_height=self.detector.pixel_height[indices]
        )
        gs = GeometrySequence(
            tube_position=self.tube_position[indices],
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
            tube_position=xp.copy(self.tube_position),
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


def normalize_geoms_(
    geometries: GeometrySequence,
    volume_extent_min,
    volume_extent_max,
    volume_voxel_size,
    volume_rotation
):
    xp = geometries.xp
    shift_(geometries,
           -(xp.asarray(volume_extent_min) + xp.asarray(volume_extent_max)) / 2)
    scale_(geometries,
           xp.asarray(volume_voxel_size))
    rotate_(geometries, *volume_rotation)


def shift(geom, shift_vector):
    geom = copy.deepcopy(geom)
    geom.tube_position += shift_vector
    geom.detector_position += shift_vector
    return geom


def shift_(geom, shift_vector: np.ndarray):
    geom.tube_position += shift_vector
    geom.detector_position += shift_vector


def scale(geom, scaling: np.ndarray):
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

    geom.tube_position[:] = geom.tube_position[:] / scaling
    geom.detector_position[:] = geom.detector_position[:] / scaling
    return geom


def scale_(geom, scaling):
    # detector pixels have to be scaled first, because
    # detector.width and detector.height need to be scaled accordingly
    xp = geom.xp
    pixel_vec = geom.u * geom.detector.pixel_width[..., xp.newaxis]  # still f32
    pixel_vec /= scaling  # keeps dtype
    geom.detector.pixel_width = xp.linalg.norm(pixel_vec, axis=-1)
    xp.divide(pixel_vec, geom.detector.pixel_width[..., xp.newaxis],
              out=geom.u)

    xp.multiply(geom.v, geom.detector.pixel_height[..., xp.newaxis],
                out=pixel_vec)
    pixel_vec /= scaling
    geom.detector.pixel_height = xp.linalg.norm(pixel_vec, axis=-1)
    xp.divide(pixel_vec, geom.detector.pixel_height[..., xp.newaxis],
              out=geom.v)
    xp.divide(geom.tube_position, scaling, out=geom.tube_position)
    xp.divide(geom.detector_position, scaling, out=geom.detector_position)


def rotate(geom,
           roll: float = 0., pitch: float = 0., yaw: float = 0.):
    ngeom = copy.deepcopy(geom)
    RT = Geometry.angles2mat(roll, pitch, yaw).T
    ngeom.tube_position = RT @ geom.tube_position
    ngeom.detector_position = RT @ geom.detector_position
    ngeom.u = RT @ geom.u
    ngeom.v = RT @ geom.v
    return ngeom


def rotate_(geom,
            roll: float = 0., pitch: float = 0., yaw: float = 0.):
    xp = geom.xp
    dtype = geom.tube_position.dtype
    R = xp.asarray(Geometry.angles2mat(roll, pitch, yaw), dtype=dtype)
    geom.tube_position[...] = geom.tube_position @ R
    geom.detector_position[...] = geom.detector_position @ R
    geom.u[...] = geom.u @ R
    geom.v[...] = geom.v @ R


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