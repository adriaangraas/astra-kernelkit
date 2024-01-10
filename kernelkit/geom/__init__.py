from typing import Any
import warnings

from kernelkit.geom.proj import *
from kernelkit.geom.vol import VolumeGeometry


def resolve_volume_geometry(
    shape=None,
    extent_min=None,
    extent_max=None,
    voxel_size=None,
    projection_geometry=None,
    verbose=False,
):
    """Resolves a `VolumeGeometry` given partial parameters.

    Parameters
    ----------
    shape : array-like, optional
        The shape of the volume, by default None.
        Pass as a scalar, or a tuple of length 3 with (x ,y, z) filled
        with None for unknown dimensions.
    extent_min : array-like, optional
        The minimum extent of the volume, by default None.
        Pass as a scalar, or a tuple of length 3 with (x ,y, z) filled
        with None for unknown dimensions.
        Pass as a scalar, or a tuple of length 3 with (x ,y, z) filled
        with None for unknown dimensions.
    extent_max : array-like, optional
        The maximum extent of the volume, by default None.
        Pass as a scalar, or a tuple of length 3 with (x ,y, z) filled
        with None for unknown dimensions.
    voxel_size : array-like, optional
        The size of a single voxel, by default None.
        Pass as a scalar, or a tuple of length 3 with (x ,y, z) filled
        with None for unknown dimensions.
    projection_geometry : ProjectionGeometry, optional
        A projection geometry, by default None. Is used to suggest the
        volume extent in case insufficient information can be deduced from
        the other parameters.
    verbose : bool, optional
        Whether to print the parameters solving process, by default False.

    Returns
    -------
    VolumeGeometry
        The resolved volume geometry.

    Notes
    -----
     - Currently only volume geometries with isotropic voxels are deduced.
     - The function may not respect the input parameters exactly, if this is
         necessary to fulfill the constraint of isotropic voxels.
    """

    def _parse(arg):
        if np.isscalar(arg) or arg is None:
            arg = [arg] * 3
        return list(arg)

    shape = _parse(shape)
    extent_min = _parse(extent_min)
    extent_max = _parse(extent_max)
    vox_sz = _parse(voxel_size)
    resolved_dims = [False] * 3

    def _solve(n, xmin, xmax, sz, dim):
        """Attempt to resolve a single dimension.

        Notes
        -----
        The function works on the variables in the outer scope.

        The following equations hold:
            nr_voxels (`n`) * voxel_size (`sz`) = volume extent (`xmax - xmin`)
        When two of these parameters are known, a third is resolved.
        """
        inferred = False
        if n is not None:
            if xmin is not None and xmax is not None:
                x = xmax - xmin
                if sz is not None:
                    if not np.allclose(n * sz, x):
                        raise ValueError(
                            f"The {dim}-th dimension contains conflicting parameter "
                            f"values: nr_voxels={n}, voxel_size={sz}, "
                            f"extent_min={xmin}, extent_max={xmax}. It must hold "
                            "that nr_voxels * voxel_size = extent_max - extent_min."
                        )
                else:
                    sz = x / n  # resolved
                inferred = True
            elif xmin is None and xmax is None:
                if sz is not None:
                    xmin = -n * sz / 2
                    xmax = n * sz / 2
                    inferred = True
                else:
                    pass  # unresolved
            else:
                raise ValueError(
                    f"Volume extent in dim {dim} must be given with both max"
                    " and min value, or inferred automatically."
                )
        else:
            if xmin is not None and xmax is not None:
                if sz is not None:
                    x = xmax - xmin
                    n = x / sz
                    if not np.allclose(n, np.round(n)):
                        raise ValueError(
                            f"Number of voxels in dim {dim} was inferred to be"
                            f" {n}, which is not a rounded number. To have "
                            " isotropic voxels, please give in larger"
                            " volume size, or leave the volume extent out in"
                            f" dim {dim} to have it inferred automatically."
                        )
                    n = int(np.round(n))
                    inferred = True
                else:
                    pass  # not inferred
            elif xmin is None and xmax is None:
                pass  # two unknowns: not inferred
            else:
                raise ValueError(
                    f"Volume extent in dim {dim} must be given with both max"
                    " and min value, or inferred automatically."
                )

        if inferred:
            if verbose:
                print(
                    "Solved:"
                    f"\n - shape: \t\t {n}"
                    f"\n - extent min: \t {xmin}"
                    f"\n - extent max: \t {xmax}"
                    f"\n - voxel size: \t {sz}"
                )
            return n, xmin, xmax, sz

        return False

    def _resolve_dims(dims, vol_ext_min, vol_ext_max):
        """Attempt to resolve a set of dimensions."""
        for d in dims:
            if resolved_dims[d]:
                continue

            if verbose:
                print(f"Attempting dimension {d}...")
            attempt = _solve(shape[d], vol_ext_min[d], vol_ext_max[d], vox_sz[d], d)
            if attempt:
                shape[d] = attempt[0]
                extent_min[d] = attempt[1]
                extent_max[d] = attempt[2]
                vox_sz[d] = attempt[3]
                resolved_dims[d] = True

    # 1: try to resolve at least one dimension, so that we have a voxel size.
    _resolve_dims(range(3), extent_min, extent_max)

    # 2: if we failed to do that, throw in automatic geometry inference
    if not np.any(resolved_dims):
        if projection_geometry is not None:
            # assume geometry is centered
            sugg_ext_min, sugg_ext_max = suggest_volume_extent(
                projection_geometry[0], (0.0, 0.0, 0)
            )

            # try to put in a known geometry value, first x, then y, then z
            for d, resolved in enumerate(resolved_dims):
                if resolved:
                    continue

                try_ext_min = np.copy(extent_min)
                try_ext_max = np.copy(extent_max)
                try_ext_min[d] = sugg_ext_min[d]
                try_ext_max[d] = sugg_ext_max[d]
                _resolve_dims([d], try_ext_min, try_ext_max)

                if resolved_dims[d]:
                    break  # we're happy with one dim, as we'd like isotropy
        else:
            # the user didn't give enough information
            raise ValueError(
                "Not enough information is provided to infer a "
                "voxel size, number of voxels and volume extent "
                "of at least a single dimension. Consider "
                "passing a geometry for automatic inference."
            )

    # 3: replicate the minimum voxel size to other dimensions, and retry
    vox_sz = np.array(vox_sz)
    vox_sz[vox_sz == None] = np.min(vox_sz[np.where(vox_sz != None)])
    vox_sz.astype(np.float32)
    _resolve_dims(range(3), extent_min, extent_max)

    # replicate x, y dims, if necessary
    if resolved_dims[0] and not resolved_dims[1]:
        shape[1] = shape[0]
    if resolved_dims[1] and not resolved_dims[0]:
        shape[0] = shape[1]
    # retry a resolve of the rest of the equations
    _resolve_dims(range(2), extent_min, extent_max)

    # if any remaining, one dim has only a voxel size, and no nr_voxels or geom
    if not np.all(resolved_dims) and projection_geometry is not None:
        # in that case, use the geom
        sugg_ext_min, sugg_ext_max = suggest_volume_extent(
            projection_geometry[0], (0.0, 0.0, 0)
        )  # assume geometry is centered

        # try to put in a known geometry value
        for d, resolved in enumerate(resolved_dims):
            if resolved:
                continue

            try_ext_min = np.copy(extent_min)
            try_ext_max = np.copy(extent_max)
            # round up geometry to have exact number of voxels
            extent = sugg_ext_max[d] - sugg_ext_min[d]
            nr_voxels_required = extent / vox_sz[d]
            # enlarge the geometry slightly, if necessary to have full voxels
            if np.allclose(np.round(nr_voxels_required), nr_voxels_required):
                nr_voxels_required = np.round(nr_voxels_required)
            else:
                nr_voxels_required = np.ceil(nr_voxels_required)
            try_ext_min[d] = -nr_voxels_required / 2 * vox_sz[d]
            try_ext_max[d] = nr_voxels_required / 2 * vox_sz[d]
            _resolve_dims([d], try_ext_min, try_ext_max)

    if verbose:
        print(
            "The final deduced volume parameters are: "
            f"\n - shape: \t\t {shape}"
            f"\n - extent min: \t {extent_min}"
            f"\n - extent max: \t {extent_max}"
            f"\n - voxel size: \t {vox_sz}"
        )

    if not np.all(resolved_dims):
        raise RuntimeError(
            "Could not resolve the volume geometry. The "
            "following dimensions cannot be inferred: "
            f"{np.where(np.logical_not(resolved_dims))[0]}. "
            "Provide two of the three unknowns and/or one "
            "of the projection geometries to deduce the "
            "dimensions."
        )

    return VolumeGeometry(
        shape=shape, voxel_size=vox_sz, extent_min=extent_min, extent_max=extent_max
    )


def suggest_volume_extent(
    geometry: ProjectionGeometry, object_position: Sequence = (0.0, 0.0, 0.0)
):
    """Suggest a volume extent for a given geometry.

    Parameters
    ----------
    geometry : ProjectionGeometry
        The geometry to suggest a volume extent for.
    object_position : Sequence, optional
        The position of the object in the volume, by default (0., 0., 0.).

    Returns
    -------
    tuple
        The suggested minimum and maximum extent of the volume.

    Notes
    -----
    In case of a large cone angle, the volume extent may be too small.
    """
    # TODO(Adriaan): now using only one geometry.
    #    perhaps use polygon clipping on multiple geoms
    #   to find the intersection of volume areas.
    g = copy.deepcopy(geometry)  # TODO
    shift_(g, -np.array(object_position))

    # assert that origin is on the source-detector line
    source_vec = g.source_position
    det_vec = g.detector_position - g.source_position
    if not np.linalg.matrix_rank([source_vec, det_vec]) == 1:
        warnings.warn(
            "Volume extents may not be suggested correctly when "
            " the geometry does not project through the origin."
        )

    SOD = np.linalg.norm(source_vec)
    SDD = np.linalg.norm(det_vec)
    # using that pixel_width / SDD = voxel_width / SOD
    # => voxel_width = pixel_width / SOD * SDD
    width = SOD * g.detector.width / SDD
    height = SOD * g.detector.height / SDD
    return (
        np.array([-width / 2, -width / 2, -height / 2]),
        np.array([width / 2, width / 2, height / 2]),
    )


def normalize_(geometries: list[ProjectionGeometry], volume_geometry: VolumeGeometry):
    """In-place transform of projections to compensate volume normalization.

    Parameters
    ----------
    geometries : list[ProjectionGeometry]
        The projection geometries to normalize.
    volume_geometry : VolumeGeometry
        The volume geometry to normalize to.

    Notes
    -----
    This function is a helper for kernels to work on a unit cube, rather than
    a volume with arbitrary extents, axis and voxel sizes.
    """
    xp = geometries.xp
    center = (
        -(
            xp.asarray(volume_geometry.extent_min)
            + xp.asarray(volume_geometry.extent_max)
        )
        / 2
    )
    shift_(geometries, center)
    scale_(geometries, xp.asarray(volume_geometry.voxel_size))
    rotate_(geometries, *volume_geometry.rotation)
