from __future__ import absolute_import, division, print_function

import copy

import numpy as np
from odl.discr import DiscretizedSpace
from odl.tomo import cone_2d_geom_to_astra_vecs
from odl.tomo.geometry import (ConeBeamGeometry, FanBeamGeometry,
                               Flat1dDetector, Flat2dDetector, Geometry)

import astrapy
import astrapy.kernel as kernel


def _cone_2d_to_geom2d(geometry):
    vecs = cone_2d_geom_to_astra_vecs(geometry, coords='ASTRA')

    geoms = list()
    for i, angle in enumerate(vecs):
        det = astrapy.geom2d.Flat1DDetector(
            geometry.detector.size,
            geometry.det_partition.cell_sides[0])
        geoms.append(astrapy.geom2d.Static2DGeometry(
            tube_pos=vecs[i, :2],
            det_pos=vecs[i, 2:4],
            det_rotation=geometry.angles[i],
            detector=det))

    return geoms


def _cone_3d_to_geom3d(geometry):
    def _compat_swap_geoms(geometries):
        for geom in geometries:
            _swp = lambda x: np.array([x[2], x[1], x[0]])
            geom.u[:] = _swp(geom.u[:])
            geom.v[:] = _swp(geom.v[:])
            geom.tube_position[:] = _swp(geom.tube_position[:])
            geom.detector_position[:] = _swp(geom.detector_position[:])

    det = astrapy.geom.Detector(
        rows=geometry.detector.shape[0],
        cols=geometry.detector.shape[1],
        pixel_width=geometry.det_partition.cell_sides[1],
        pixel_height=geometry.det_partition.cell_sides[0])

    geoms = list()
    for angle in zip(geometry.angles):
        det_axes = np.moveaxis(geometry.det_axes(angle), -2, 0)
        geoms.append(astrapy.geom.Geometry(
            tube_pos=geometry.src_position(angle)[0],
            det_pos=geometry.det_point_position(
                angle, geometry.det_params.mid_pt)[0],
            u_unit=det_axes[0][0],
            v_unit=det_axes[1][0],
            detector=copy.deepcopy(det)))

    _compat_swap_geoms(geoms)
    return geoms


class RayTrafoImpl:
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
        self.filter = False

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

        # sino = SinogramAdapter(self.geometry)

        if (isinstance(self.geometry, FanBeamGeometry)
            and isinstance(self.geometry.detector, Flat1dDetector)
            and self.geometry.ndim == 2):
            fp = kernel.FanProjection()
            geom = _cone_2d_to_geom2d(self.geometry)
            fp(volume, sino, geom)
        elif (isinstance(self.geometry, ConeBeamGeometry)
              and isinstance(self.geometry.detector, Flat2dDetector)
              and self.geometry.ndim == 3):
            geom = _cone_3d_to_geom3d(self.geometry)
            sino = astrapy.fp(
                vol_data.data,
                geom,
                volume_extent_min=self.vol_space.min_pt,
                volume_extent_max=self.vol_space.max_pt,
                chunk_size=1200)

            # TODO: issue #2
            out[:] = sino
        else:
            raise NotImplementedError(
                'Unknown AstraPy geometry type {!r}, incorrect detector,'
                ' or incorrect geometry dimension. '.format(self.geometry))

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

        if (isinstance(self.geometry, FanBeamGeometry)
            and isinstance(self.geometry.detector, Flat1dDetector)
            and self.geometry.ndim == 2):
            geom = _cone_2d_to_geom2d(self.geometry)
            bp = kernel.FanBackprojection()
            bp(volume, sino, geom, volume_shape)
        elif (isinstance(self.geometry, ConeBeamGeometry)
              and isinstance(self.geometry.detector, Flat2dDetector)
              and self.geometry.ndim == 3):
            geom = _cone_3d_to_geom3d(self.geometry)
            volume = astrapy.bp(proj_data.data,
                                geom,
                                volume_shape=self.vol_space.shape,
                                volume_extent_min=self.vol_space.min_pt,
                                volume_extent_max=self.vol_space.max_pt,
                                chunk_size=300,
                                filter=None if not self.filter else 'ram-lak')
        else:
            raise NotImplementedError(
                'Unknown AstraPy geometry type {!r}, incorrect detector,'
                ' or incorrect geometry dimension. '.format(self.geometry))

        # Copy result to CPU memory, TODO: issue #2 again
        out[:] = volume

        # no more scaling here?
        # TODO: weighted spaces, how to handle them
        return out
