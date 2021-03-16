from __future__ import absolute_import, division, print_function

import copy
from builtins import object

import numpy as np
from odl.discr import DiscretizedSpace, DiscretizedSpaceElement
from odl.tomo import cone_2d_geom_to_astra_vecs
from odl.tomo.geometry import (ConeBeamGeometry, FanBeamGeometry,
                               Flat1dDetector, Flat2dDetector, Geometry)

import astrapy
import astrapy.kernel as kernel


class VolumeAdapter(astrapy.Volume):
    """Expose a DiscretizedSpace element as an ASTRApy volume."""

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
            raise TypeError(
                '`vol_space` {!r} is not a DiscretizedSpace instance'
                ''.format(vol_space))

        if not vol_space.is_uniform:
            raise ValueError('`vol_space` {} is not uniformly discretized')

        vol_min = vol_space.partition.min_pt
        vol_max = vol_space.partition.max_pt

        # TODO: how does the axis order vary in ODL?
        vol_data_numpy = vol_data.asarray()
        super().__init__(vol_data_numpy, vol_min, vol_max)
        self.vol_space = vol_space

    @property
    def has_isotropic_voxels(self) -> bool:
        return self.vol_space.partition.has_isotropic_cells

    @property
    def voxel_volume(self) -> float:
        return self.vol_space.partition.cell_volume


class SinogramAdapter(astrapy.Sinogram):
    """Expose a DiscretizedSpace element as an ASTRApy sinogram."""

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
        if not isinstance(geometry, Geometry):
            raise TypeError('`geometry` {!r} is not a FanBeamGeometry instance'
                            ''.format(geometry))

        if proj_data is not None:
            proj_space = proj_data.space

            # TODO: how to check if this is consistent with the geometry?

            if not isinstance(proj_space, DiscretizedSpace):
                raise TypeError(
                    '`proj_space` {!r} is not a DiscretizedSpace instance'
                    ''.format(proj_space))

            if not proj_space.is_uniform:
                raise ValueError(
                    '`proj_space` {} is not uniformly discretized')

            data = proj_data.asarray()
        else:
            data = astrapy.zeros_gpu(geometry.grid.shape,
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


def _cone_2d_to_geom2d(geometry):
    vecs = cone_2d_geom_to_astra_vecs(geometry, coords='ASTRA')

    geoms = list()
    for i, angle in enumerate(vecs):
        det = astrapy.Flat1DDetector(
            geometry.detector.size,
            geometry.det_partition.cell_sides[0])
        geoms.append(astrapy.Static2DGeometry(
            tube_pos=vecs[i, :2],
            det_pos=vecs[i, 2:4],
            det_rotation=geometry.angles[i],
            detector=det))

    return geoms


def _cone_3d_to_geom3d(geometry):
    angles = geometry.angles
    mid_pt = geometry.det_params.mid_pt
    vectors = np.zeros((angles.size, 12))

    # Source position
    vectors[:, 0:3] = geometry.src_position(angles)

    # Center of detector in 3D space
    vectors[:, 3:6] = geometry.det_point_position(angles, mid_pt)
    det_axes = np.moveaxis(geometry.det_axes(angles), -2, 0)
    vectors[:, 6:9] = det_axes[0]
    vectors[:, 9:12] = det_axes[1]

    det = astrapy.Flat2DDetector(
        rows=geometry.detector.shape[1],
        cols=geometry.detector.shape[0],
        pixel_width=geometry.det_partition.cell_sides[0],
        pixel_height=geometry.det_partition.cell_sides[1])

    geoms = list()
    for i, angle in enumerate(vectors):
        geoms.append(astrapy.AstraStatic3DGeometry(
            tube_pos=vectors[i, :3],
            det_pos=vectors[i, 3:6],
            u_unit=vectors[i, 6:9],
            v_unit=vectors[i, 9:12],
            detector=copy.deepcopy(det)))

    return geoms


class RayTrafoImpl(object):
    """Thin wrapper around ASTRApy."""

    def __init__(self, geometry, vol_space, proj_space):
        """Initialize a new instance.

        TODO(Adriaan): constrain ODL version

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

        volume = VolumeAdapter(vol_data)
        sino = SinogramAdapter(self.geometry)

        if (isinstance(self.geometry, FanBeamGeometry)
            and isinstance(self.geometry.detector, Flat1dDetector)
            and self.geometry.ndim == 2):
            fp = kernel.FanProjection()
            geom = _cone_2d_to_geom2d(self.geometry)
            fp(volume, sino, geom)
        elif (isinstance(self.geometry, ConeBeamGeometry)
              and isinstance(self.geometry.detector, Flat2dDetector)
              and self.geometry.ndim == 3):
            fp = kernel.ConeProjection()
            geom = _cone_3d_to_geom3d(self.geometry)
            fp(volume, sino, geom)
        else:
            raise NotImplementedError(
                'Unknown AstraPy geometry type {!r}, incorrect detector,'
                ' or incorrect geometry dimension. '.format(self.geometry))

        out[:] = sino.data_numpy  # TODO: issue #2
        return out

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

        # TODO(Adriaan): Use `out` as volume?
        if out is not None:
            assert out in self.vol_space
        else:
            out = self.vol_space.element()

        volume = astrapy.Volume(
            data=astrapy.zeros_gpu(
                self.vol_space.shape,
                dtype=kernel.Kernel.FLOAT_DTYPE
            ),
            extent_min=self.vol_space.min_pt,
            extent_max=self.vol_space.max_pt)
        sino = SinogramAdapter(self.geometry, proj_data)

        if (isinstance(self.geometry, FanBeamGeometry)
            and isinstance(self.geometry.detector, Flat1dDetector)
            and self.geometry.ndim == 2):
            geom = _cone_2d_to_geom2d(self.geometry)
            bp = kernel.FanBackprojection()
            bp(volume, sino, geom)
        elif (isinstance(self.geometry, ConeBeamGeometry)
              and isinstance(self.geometry.detector, Flat2dDetector)
              and self.geometry.ndim == 3):
            geom = _cone_3d_to_geom3d(self.geometry)
            bp = kernel.ConeBackprojection()
            bp(volume, sino, geom)
        else:
            raise NotImplementedError(
                'Unknown AstraPy geometry type {!r}, incorrect detector,'
                ' or incorrect geometry dimension. '.format(self.geometry))

        # Copy result to CPU memory, TODO: issue #2 again
        out[:] = volume.data_numpy

        # no more scaling here?
        # TODO: weighted spaces, how to handle them
        return out
