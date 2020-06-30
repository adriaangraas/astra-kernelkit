# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Backend for ASTRApy."""

from __future__ import absolute_import, division, print_function

from builtins import object

import numpy as np
from odl.discr import DiscretizedSpace, DiscretizedSpaceElement
from odl.tomo.geometry import (DivergentBeamGeometry, FanBeamGeometry,
                               Flat1dDetector, Flat2dDetector, Geometry)
from odl.tomo.util.utility import euler_matrix

import astrapy
import astrapy.kernel as kernel


class VolumeAdapter(astrapy.Volume):
    """Wrapper for a DiscretizedSpaceElement to interface with ASTRApy volume."""

    def __init__(self, vol_data: DiscretizedSpaceElement):
        """Create an ASTRApy Volume

        Parameters
        ----------
        vol_space : `DiscretizedSpace`
            Discretized space where the reconstruction (volume) lives.
            It must be 2- or 3-dimensional and uniformly discretized.
        """
        vol_space = vol_data.space

        if not isinstance(vol_space, DiscretizedSpace):
            raise TypeError('`vol_space` {!r} is not a DiscretizedSpace instance'
                            ''.format(vol_space))

        if not vol_space.is_uniform:
            raise ValueError('`vol_space` {} is not uniformly discretized')

        vol_min = vol_space.partition.min_pt
        vol_max = vol_space.partition.max_pt

        # @todo how does the axis order vary in ODL?
        super().__init__(vol_data.asarray(), vol_min, vol_max)
        self.vol_space = vol_space

    @property
    def has_isotropic_voxels(self) -> bool:
        return self.vol_space.partition.has_isotropic_cells

    @property
    def voxel_volume(self) -> float:
        return self.vol_space.partition.cell_volume


class SinogramAdapter(astrapy.Sinogram):
    """Wrapper for a DiscretizedSpace element to interface with ASTRApy sinogram."""

    def __init__(self, geometry: Geometry,
                 proj_data: DiscretizedSpaceElement = None):
        """Create an ASTRApy Volume

        Parameters
        ----------
        geometry : `Geometry`
        proj_data : `DiscretizedSpace`
            Discretized space where the projection data (volume) lives.
            It must be time + 2- or 3-dimensional and uniformly discretized.
            If `None` is given then `data` will auto-initialize an empty array.
        """
        if not isinstance(geometry, FanBeamGeometry):
            raise TypeError('`geometry` {!r} is not a FanBeamGeometry instance'
                            ''.format(geometry))

        if proj_data is not None:
            proj_space = proj_data.space

            # todo: how to check if this is consistent with the geometry?

            if not isinstance(proj_space, DiscretizedSpace):
                raise TypeError(
                    '`proj_space` {!r} is not a DiscretizedSpace instance'
                    ''.format(proj_space))

            if not proj_space.is_uniform:
                raise ValueError(
                    '`proj_space` {} is not uniformly discretized')

            data = proj_data.asarray()
        else:
            data = astrapy.empty_gpu(geometry.grid.shape,
                                     dtype=astrapy.kernel.Kernel.FLOAT_DTYPE)

        proj_min = geometry.det_partition.min_pt
        proj_max = geometry.det_partition.max_pt

        super().__init__(data, proj_min, proj_max)
        # self.proj_space = proj_space
        self.geometry = geometry

    @property
    def has_isotropic_pixels(self) -> bool:
        return self.geometry.det_partition.has_isotropic_cells

    @property
    def pixel_volume(self) -> float:
        return self.geometry.det_partition.cell_volume


class GeometryDictAdapter(astrapy.GeometryDict):
    def __init__(self, geometry: DivergentBeamGeometry):
        super().__init__()
        self.geometry = geometry

        # Instead of rotating the data by 90 degrees counter-clockwise,
        # we subtract pi/2 from the geometry angles, thereby rotating the
        # geometry by 90 degrees clockwise
        rot_minus_90 = euler_matrix(-np.pi / 2)
        angles = geometry.angles  # type: FanBeamGeometry
        vectors = np.zeros((angles.size, 6))

        # Source position
        src_pos = geometry.src_position(angles)
        vectors[:, 0:2] = rot_minus_90.dot(src_pos.T).T  # dot along 2nd axis

        # Center of detector
        mid_pt = geometry.det_params.mid_pt
        # Need to cast `mid_pt` to float since otherwise the empty axis is
        # not removed
        centers = geometry.det_point_position(angles, float(mid_pt))
        vectors[:, 2:4] = rot_minus_90.dot(centers.T).T

        # Vector from detector pixel 0 to 1
        det_axis = rot_minus_90.dot(geometry.det_axis(angles).T).T
        px_size = geometry.det_partition.cell_sides[0]
        vectors[:, 4:6] = det_axis * px_size

        # (srcX, srcY, dX, dY, uX, uY) -> (dX, dY, ..., srcX, srcY)
        vec = np.roll(vectors, -2, axis=1)

        for i, angle in enumerate(angles):
            det = astrapy.Flat1DDetector(geometry.detector.size,
                                         geometry.det_partition.cell_sides[0])
            self._geoms_at_times[i] = astrapy.Static2DGeometry(vec[i, :2],
                                                               vec[i, 2:4],
                                                               geometry.angles[
                                                                   i], det)


class RayTrafoImpl(object):
    """Thin wrapper around ASTRApy."""

    def __init__(self, geometry, vol_space, proj_space):
        """Initialize a new instance.

        Parameters
        ----------
        geometry : `Geometry`
            Geometry defining the tomographic setup.
        vol_space : `DiscretizedSpace`
            Reconstruction space, the space of the images to be forward
            projected.
        proj_space : `DiscretizedSpace`
            Projection space, the space of the result.
        """
        if not isinstance(geometry, Geometry):
            raise TypeError('`geometry` {!r} is not a `Geometry` instance'
                            ''.format(geometry))

        assert isinstance(vol_space, DiscretizedSpace)
        assert isinstance(proj_space, DiscretizedSpace)

        self.geometry = geometry
        self.vol_space = vol_space
        self.proj_space = proj_space

    def call_forward(self, vol_data, out=None):
        """Run an ASTRA forward projection on the given data

        Parameters
        ----------
        vol_data : ``vol_space`` element
            Volume data to which the projector is applied.
        out : ``proj_space`` element, optional
            Element of the projection space to which the result is written. If
            ``None``, an element in `proj_space` is created.

        Returns
        -------
        out : ``proj_space`` element
            Projection data resulting from the application of the projector.
            If ``out`` was provided, the returned object is a reference to it.
        """
        assert vol_data in self.vol_space
        if out is not None:
            assert out in self.proj_space
        else:
            out = self.proj_space.element()

        if not self.geometry.det_partition.is_uniform:
            raise ValueError('Non-uniform detector sampling is not supported')

        if (isinstance(self.geometry, DivergentBeamGeometry) and
            isinstance(self.geometry.detector,
                       (Flat1dDetector, Flat2dDetector)) and
            self.geometry.ndim == 2):
            fp = kernel.FanProjection()
            geom = GeometryDictAdapter(self.geometry)
            volume = VolumeAdapter(vol_data)
            sino = SinogramAdapter(self.geometry)

            fp(volume, sino, geom)
        else:
            raise NotImplementedError(
                'Unknown Astrapy geometry type {!r}'.format(self.geometry))

        # Copy result to host
        if self.geometry.ndim == 2:
            out[:] = sino.data_numpy  # @todo issue #2
        elif self.geometry.ndim == 3:
            raise NotImplementedError()

        return out


class AstrapyBackProjectorImpl(object):
    """Thin wrapper around ASTRA."""

    def __init__(self, geometry, vol_space, proj_space):
        """Initialize a new instance.

        Parameters
        ----------
        geometry : `Geometry`
            Geometry defining the tomographic setup.
        vol_space : `DiscretizedSpace`
            Reconstruction space, the space to which the backprojection maps.
        proj_space : `DiscretizedSpace`
            Projection space, the space from which the backprojection maps.
        """

        if not isinstance(geometry, Geometry):
            raise TypeError('`geometry` {!r} is not a `Geometry` instance'
                            ''.format(geometry))

        assert isinstance(vol_space, DiscretizedSpace)
        assert isinstance(proj_space, DiscretizedSpace)

        self.geometry = geometry
        self.vol_space = vol_space
        self.proj_space = proj_space

    def call_backward(self, proj_data, out=None):
        """Run an ASTRA back-projection on the given data using the GPU.

        Parameters
        ----------
        proj_data : ``proj_space`` element
            Projection data to which the back-projector is applied.
        out : ``vol_space`` element, optional
            Element of the reconstruction space to which the result is written.
            If ``None``, an element in ``vol_space`` is created.

        Returns
        -------
        out : ``vol_space`` element
            Reconstruction data resulting from the application of the
            back-projector. If ``out`` was provided, the returned object is a
            reference to it.
        """
        assert proj_data in self.proj_space

        if out is not None:
            assert out in self.vol_space
        else:
            out = self.vol_space.element()

        if self.geometry.motion_partition.ndim == 1:
            motion_shape = self.geometry.motion_partition.shape
        else:
            # Need to flatten 2- or 3-dimensional angles into one axis
            motion_shape = (np.prod(self.geometry.motion_partition.shape),)

        if len(motion_shape + self.geometry.det_partition.shape) != 2:
            raise NotImplementedError(
                "3D is not yet implemented, but we would need to take care with axis here.")

        sino = SinogramAdapter(self.geometry, proj_data)
        angles = astrapy_fanflat_geometry_to_angles(self.geometry)

        volume_data = astrapy.empty_gpu(self.vol_space.shape,
                                        dtype=kernel.Kernel.FLOAT_DTYPE)
        volume = astrapy.Volume(volume_data, self.vol_space.min_pt,
                                self.vol_space.max_pt)

        bp = kernel.FanBackprojection()
        volume = bp(volume, sino, angles)

        # Copy result to CPU memory, todo: issue #2 again
        out[:] = volume.data_numpy

        # no more scaling here?
        # @todo weighted spaces, how to handle them
        return out
