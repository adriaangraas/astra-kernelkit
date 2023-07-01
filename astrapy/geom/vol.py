from typing import Tuple

import numpy as np


def _voxel_size(volume_shape,
               extent_min,
               extent_max) -> Tuple[float, float, float]:
    """The physical size of a voxel."""
    n = np.array(volume_shape)
    dists = np.array(extent_max) - np.array(extent_min)
    shp = list(dists / n)
    return tuple(np.array(shp))


class VolumeGeometry:
    def __init__(self,
                 shape,
                 voxel_size,
                 extent_min,
                 extent_max,
                 rotation=(0., 0., 0.)):
        self.shape = tuple(np.array(shape, dtype=np.int32))
        self.voxel_size = tuple(np.array(voxel_size, dtype=np.float32))
        self.extent_min = tuple(np.array(extent_min, dtype=np.float32))
        self.extent_max = tuple(np.array(extent_max, dtype=np.float32))
        self.rotation = tuple(np.array(rotation, dtype=np.float32))

    def has_isotropic_voxels(self, atol: float = 1e-8) -> bool:
        return np.allclose(self.voxel_size, self.voxel_size[0], atol=atol)

    @property
    def voxel_volume(self) -> float:
        vox_size = _voxel_size(self.shape, self.extent_min, self.extent_max)
        return float(np.prod(vox_size))