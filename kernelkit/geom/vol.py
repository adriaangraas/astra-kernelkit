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

    def __init__(
        self, shape, voxel_size, extent_min, extent_max,
        rotation=(0.0, 0.0, 0.0)
    ):
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
               kernelkit.geom.vol.resolve_volume_geometry`.

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

    def check_consistency(self) -> bool:
        """Checks if the volume geometry is consistent.

        Raises
        ------
        ValueError
            If the volume geometry is not consistent.

        Returns
        -------
        bool
            True if the volume geometry is consistent.
        """
        # Check dimensions of shape, voxel_size, extent_min, extent_max, rotation
        if len(self.shape) not in [2, 3]:
            raise ValueError(
                f"Shape must have 2 or 3 dimensions, got {len(self.shape)}.")
        if len(self.voxel_size) != len(self.shape):
            raise ValueError(f"Voxel size must match the shape dimensions, "
                             f"got {len(self.voxel_size)} dimensions.")
        if len(self.extent_min) != len(self.shape):
            raise ValueError(f"Extent min must match the shape dimensions, "
                             f"got {len(self.extent_min)} dimensions.")
        if len(self.extent_max) != len(self.shape):
            raise ValueError(f"Extent max must match the shape dimensions, "
                             f"got {len(self.extent_max)} dimensions.")
        if len(self.rotation) != 3:
            raise ValueError(f"Rotation must have 3 dimensions, "
                             f"got {len(self.rotation)} dimensions.")

        # Ensure extent_min is less than extent_max
        for min_val, max_val in zip(self.extent_min, self.extent_max):
            if min_val >= max_val:
                raise ValueError(f"Extent min {min_val} must be "
                                 f"less than extent max {max_val}.")

        # Ensure the volume described by shape, voxel_size, extent_min,
        # and extent_max is non-negative
        for dim_size, voxel_size in zip(self.shape, self.voxel_size):
            if dim_size <= 0:
                raise ValueError(f"Shape dimension {dim_size} "
                                 f"must be positive.")
            if voxel_size <= 0:
                raise ValueError(f"Voxel size {voxel_size} "
                                 f"must be positive.")

        # Check that the number of voxels times voxel size matches the extents
        for dim_size, voxel_size, min_val, max_val in zip(
            self.shape, self.voxel_size, self.extent_min, self.extent_max):
            expected_extent = dim_size * voxel_size
            actual_extent = max_val - min_val
            if not np.isclose(expected_extent, actual_extent):
                raise ValueError(f"Expected extent {expected_extent} does "
                                 f"not match actual extent {actual_extent} "
                                 f"for dimension.")

        return True
