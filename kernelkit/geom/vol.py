import numpy as np


def _voxel_size(shape, extent_min, extent_max) -> tuple[float, float, float]:
    """Calculates the physical size of a voxel from the volume extents.

    Parameters
    ----------
    shape : array-like
        The shape of the volume.
    extent_min : array-like
        The minimum extent of the volume.
    extent_max : array-like
        The maximum extent of the volume.

    Returns
    -------
    tuple
        The physical size of a voxel.
    """
    L = [(extent_max[i] - extent_min[i]) / shape[i] for i in range(3)]
    return tuple(L)


class VolumeGeometry:
    """Geometry for a single 3D reconstruction object."""

    def __init__(self, shape, voxel_size, extent_min, extent_max,
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
            (0., 0., 0.). The rotation is equivalent to a rotation of
            all projection geometriess in the opposite direction.

        Notes
        -----
         - To encode a shift, use the `extent_min` and `extent_max` parameters.
         - The `shape`, `voxel_size`, `extent_min`, `extent_max`, and `rotation`
           parameters are all stored as tuples. They contain redundant information,
           to generate this object from only a few parameters, see
          `kernelkit.geom.vol.resolve_volume_geometry`.
        """
        self.shape = tuple(np.array(shape, dtype=np.int32))
        self.voxel_size = tuple(np.array(voxel_size, dtype=np.float32))
        self.extent_min = tuple(np.array(extent_min, dtype=np.float32))
        self.extent_max = tuple(np.array(extent_max, dtype=np.float32))
        self.rotation = tuple(np.array(rotation, dtype=np.float32))

    def has_isotropic_voxels(self, atol: float = 1e-8) -> bool:
        """Checks if the volume has an isotropic voxel size.

        Parameters
        ----------
        atol : float, optional
            The absolute tolerance for the comparison, by default 1e-8.

        Returns
        -------
        bool
            True if the volume has isotropic voxels, False otherwise.
        """
        return np.allclose(self.voxel_size, self.voxel_size[0], atol=atol)

    def voxel_volume(self) -> float:
        """Returns the volume of a single voxel.

        Returns
        -------
        float
            The volume of a single voxel.
        """
        vox_size = _voxel_size(self.shape, self.extent_min, self.extent_max)
        # note: less concise but this is faster than np.prod(...)
        return float(vox_size[0] * vox_size[1] * vox_size[2])
