from typing import Tuple

import numpy as np


def _voxel_size(shape,
                extent_min,
                extent_max) -> Tuple[float, float, float]:
    """Calculates the physical size of a voxel.

    Parameters
    ----------
    shape : array-like
        The shape of the volume.
    extent_min : array-like
        The minimum extent of the volume.
    extent_max : array-like
        The maximum extent of the volume.
    """
    n = np.array(shape)
    dists = np.array(extent_max) - np.array(extent_min)
    shp = list(dists / n)
    return tuple(np.array(shp))


class VolumeGeometry:
    """Minimalist class for storing the geometry of a reconstruction volume.

    See Also
    --------
    astrapy.geom.vol.resolve_volume_geometry
    """

    def __init__(self,
                 shape,
                 voxel_size,
                 extent_min,
                 extent_max,
                 rotation=(0., 0., 0.)):
        """Initializes a `VolumeGeometry` object.

        Parameters
        ----------
        shape : array-like
            The shape of the volume.
        voxel_size : array-like
            The size of a single voxel.
        extent_min : array-like
            The minimum extent of the volume.
        extent_max : array-like
            The maximum extent of the volume.
        rotation : array-like, optional
            The rotation of the volume in roll, pitch, yaw order, by default
            (0., 0., 0.).

        Notes
        -----
        The `shape`, `voxel_size`, `extent_min`, `extent_max`, and `rotation`
        parameters are all stored as tuples. They contain rundant information,
        to generate this object from only a few parameters, see
        `astrapy.geom.vol.resolve_volume_geometry`.
        """
        self.shape = tuple(np.array(shape, dtype=np.int32))
        self.voxel_size = tuple(np.array(voxel_size, dtype=np.float32))
        self.extent_min = tuple(np.array(extent_min, dtype=np.float32))
        self.extent_max = tuple(np.array(extent_max, dtype=np.float32))
        self.rotation = tuple(np.array(rotation, dtype=np.float32))

    def has_isotropic_voxels(self, atol: float = 1e-8) -> bool:
        """Checks if the volume has isotropic voxels.

        Parameters
        ----------
        atol : float, optional
            The absolute tolerance for the comparison, by default 1e-8."""
        return np.allclose(self.voxel_size, self.voxel_size[0], atol=atol)

    @property
    def voxel_volume(self) -> float:
        """Returns the volume of a voxel."""
        vox_size = _voxel_size(self.shape, self.extent_min, self.extent_max)
        return float(np.prod(vox_size))
