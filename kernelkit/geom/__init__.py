import copy
import warnings
from typing import Any, Sequence

import numpy as np

from astrapy.geom.proj import (GeometrySequence, ProjectionGeometry,
                               rotate, scale, shift,
                               rotate_, scale_, shift_)
from astrapy.geom.vol import VolumeGeometry


def resolve_volume_geometry(
    shape=None,
    extent_min=None,
    extent_max=None,
    voxel_size=None,
    projection_geometry=None,
    verbose=False,
):
    """Resolves a `VolumeGeometry` given a few volume parameters. """

    def _parse(arg):
        if np.isscalar(arg) or arg is None:
            arg = [arg] * 3
        return list(arg)

    shape = _parse(shape)
    extent_min = _parse(extent_min)
    extent_max = _parse(extent_max)
    vox_sz = _parse(voxel_size)

    def solve(n, xmin, xmax, sz, dim):
        """
        Solve, for a dimension `d`:
            nr_voxels (`n`) * voxel_size (`sz`) = volume extent (`xmax - xmin`)
        When two of these parameters are unknown, the third is resolved.
        """
        inferred = False
        if n is not None:
            if xmin is not None and xmax is not None:
                x = xmax - xmin
                if sz is not None:
                    assert np.allclose(n * sz, x), (
                        f"{n} voxels * {sz} voxel_size must equal extent"
                        f" in dim {dim}.")
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
                    " and min value, or inferred automatically.")
        else:
            if xmin is not None and xmax is not None:
                if sz is not None:
                    x = xmax - xmin
                    n = x / sz
                    if not np.allclose(n, np.round(n)):
                        raise ValueError(
                            f"Number of voxels in dim {dim} was inferred to be"
                            f" {n}, which is not a rounded number. To have "
                            f" isotropic voxels, please give in larger"
                            f" volume size, or leave the volume extent out in"
                            f" dim {dim} to have it inferred automatically.")
                    n = int(np.round(n))
                    inferred = True
                else:
                    pass  # not inferred
            elif xmin is None and xmax is None:
                pass  # two unknowns: not inferred
            else:
                raise ValueError(
                    f"Volume extent in dim {dim} must be given with both max"
                    " and min value, or inferred automatically.")

        if inferred:
            # if verbose:
            #     print("Computed volume parameters: ")
            #     print(f" - shape: {shape}")
            #     print(f" - extent min: {extent_min}")
            #     print(f" - extent max: {extent_max}")
            #     print(f" - voxel size: {vox_sz}")

            return n, xmin, xmax, sz

        return False

    resolved_dims = [False] * 3

    def _resolve_dims(dims, vol_ext_min, vol_ext_max):
        for d in dims:
            if resolved_dims[d]:
                continue
            attempt = solve(shape[d], vol_ext_min[d], vol_ext_max[d],
                            vox_sz[d], d)
            if attempt:
                shape[d] = attempt[0]
                extent_min[d] = attempt[1]
                extent_max[d] = attempt[2]
                vox_sz[d] = attempt[3]
                resolved_dims[d] = True

    # try to resolve at least one dimension, so that we have a voxel size
    _resolve_dims(range(3), extent_min, extent_max)

    # if we failed to do that, throw in automatic geometry inference
    if not np.any(resolved_dims):
        if projection_geometry is not None:
            # assume geometry is centered
            sugg_ext_min, sugg_ext_max = suggest_volume_extent(
                projection_geometry[0], (0., 0., 0))

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
            raise ValueError("Not enough information is provided to infer a "
                             "voxel size, number of voxels and volume extent "
                             "of at least a single dimension. Consider "
                             "passing a geometry for automatic inference.")

    # replicate the minimum voxel size to other dimensions, and retry
    vox_sz = np.array(vox_sz)
    vox_sz[vox_sz == None] = (np.min(vox_sz[np.where(vox_sz != None)]))
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
            projection_geometry[0], (0., 0., 0))  # assume geometry is centered

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
            try_ext_min[d] = - nr_voxels_required / 2 * vox_sz[d]
            try_ext_max[d] = nr_voxels_required / 2 * vox_sz[d]
            _resolve_dims([d], try_ext_min, try_ext_max)

    if not np.all(resolved_dims):
        raise RuntimeError("Could not resolve the volume geometry. The "
                            "following dimensions are still unknown: "
                            f"{np.where(np.logical_not(resolved_dims))[0]}")


    if verbose:
        print("Computed volume parameters: ")
        print(f" - shape \t\t {shape}")
        print(f" - extent min \t {extent_min}")
        print(f" - extent max \t {extent_max}")
        print(f" - voxel size \t {vox_sz}")

    return VolumeGeometry(
        shape=shape,
        voxel_size=vox_sz,
        extent_min=extent_min,
        extent_max=extent_max)


def suggest_volume_extent(geometry: ProjectionGeometry,
                          object_position: Sequence = (0., 0., 0.)):
    """Suggest a volume extent for a given geometry."""
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
            " the geometry does not project through the origin.")

    SOD = np.linalg.norm(source_vec)
    SDD = np.linalg.norm(det_vec)
    # using that pixel_width / SDD = voxel_width / SOD
    # => voxel_width = pixel_width / SOD * SDD
    width = SOD * g.detector.width / SDD
    height = SOD * g.detector.height / SDD
    return (np.array([-width / 2, -width / 2, -height / 2]),
            np.array([width / 2, width / 2, height / 2]))


def normalize_(geometries: Any,
               volume_geometry: VolumeGeometry):
    xp = geometries.xp
    shift_(geometries, -(xp.asarray(volume_geometry.extent_min)
                         + xp.asarray(volume_geometry.extent_max)) / 2)
    scale_(geometries, xp.asarray(volume_geometry.voxel_size))
    rotate_(geometries, *volume_geometry.rotation)
