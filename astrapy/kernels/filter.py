"""
-----------------------------------------------------------------------
Copyright: 2010-2018, imec Vision Lab, University of Antwerp
           2014-2018, CWI, Amsterdam

Contact: astra@astra-toolbox.com
Website: http://www.astra-toolbox.com/

This file is part of the ASTRA Toolbox.


The ASTRA Toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The ASTRA Toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------------------------
"""
import warnings
from typing import Any, Sequence

import cupy as cp
import numpy as np

from astrapy.kernel import Kernel
from astrapy.kernels.conebeam import _shift_and_scale


class Filter(Kernel):
    ANGLES_PER_WEIGHT_BLOCK = 16
    DET_BLOCK_U = 32
    DET_BLOCK_V = 32

    def __init__(self, path: str = "../cuda/3d/filter.cu",
                 *args, **kwargs):
        super().__init__(path, *args, **kwargs)

    def _compile(self):
        module = self.load_module(
            det_block_u=self.DET_BLOCK_U,
            det_block_v=self.DET_BLOCK_V,
            angles_per_weight_block=self.ANGLES_PER_WEIGHT_BLOCK)
        return module, {'preweight': module.get_function("preweight")}

    def _filter(self, projections):
        ramlak = cp.linspace(-1, 1, num=projections[0].shape[1] // 2 + 1)
        ramlak = cp.abs(ramlak)
        for row in range(projections[0].shape[0]):
            projrows = cp.array([p[row] for p in projections])
            fourier_projs = cp.fft.fftshift(cp.fft.rfft(projrows))
            fourier_projs *= ramlak  # complex mult
            out = cp.fft.irfft(cp.fft.ifftshift(fourier_projs))
            for p, outrow in zip(projections, out):
                p[row] = outrow

    def __call__(self,
                 projections: list,
                 geoms: list,
                 filter: Any,
                 volume_extent_min: Sequence,
                 volume_extent_max: Sequence,
                 voxel_size: Sequence,
                 short_scan: bool = False):
        for proj in projections:
            if not isinstance(proj, cp.ndarray):
                raise TypeError("`projections` must be a CuPy ndarray.")

        # NB: We don't support arbitrary cone_vec geometries here.
        # Only those that are vertical sub-geometries
        # (cf. CompositeGeometryManager) of regular cone geometries.
        assert len(geoms) > 0

        # # TODO: to do preweighting I need volume info here.
        import copy
        geoms = copy.deepcopy(geoms)
        [_shift_and_scale(g,
                          volume_extent_min,
                          volume_extent_max,
                          voxel_size)
         for g in geoms]

        # TODO(Adriaan): assert geometry consistency
        g0 = geoms[0]
        nr_cols = g0.detector.cols
        nr_rows = g0.detector.rows
        nr_geoms = len(geoms)

        # TODO(Adriaan): assert that geometry works well enough with FDK?
        #  I guess this will do for now
        if g0.u[2] != 0:
            warnings.warn("Filter only works for geometries "
                          "in the horizontal plane.")
        # TODO(Adriaan): this feels sketchy
        # assuming U is in the XY plane, V is parallel to Z axis
        det_cx, det_cy = (g0.detector_position
                          + .5 * g0.u * g0.detector.width)[:2]
        tube_origin = np.linalg.norm(g0.tube_position[:2])
        det_origin = np.linalg.norm(np.array([det_cx, det_cy]))
        det_cz = (g0.detector_position + .5 * g0.v[2] * g0.detector.height)[2]
        z_shift = det_cz - g0.tube_position[2]

        # TODO(ASTRA): FIXME: Sign/order
        angles = [-np.arctan2(g.tube_position[0], g.tube_position[1]) + np.pi
                  for g in geoms]

        # The pre-weighting factor for a ray is the cosine of the angle between
        # the central line and the ray.
        blocks_u = (nr_cols + self.DET_BLOCK_U - 1) // self.DET_BLOCK_U
        blocks_v = (nr_rows + self.DET_BLOCK_V - 1) // self.DET_BLOCK_V
        dimgrid_x = blocks_u * blocks_v
        dimgrid_y = (
            (nr_geoms + self.ANGLES_PER_WEIGHT_BLOCK - 1)
            // self.ANGLES_PER_WEIGHT_BLOCK)

        module, funcs = self._get_compilation()

        proj_pointers = cp.array([p.data.ptr for p in projections])

        funcs['preweight'](
            (dimgrid_x, dimgrid_y),
            (self.DET_BLOCK_U, self.ANGLES_PER_WEIGHT_BLOCK),
            (proj_pointers,
             0,
             len(angles),
             cp.float32(tube_origin),
             cp.float32(det_origin),
             cp.float32(z_shift),
             cp.float32(g0.detector.pixel_width),
             cp.float32(g0.detector.pixel_height),
             nr_geoms,
             nr_cols,
             nr_rows))

        cp.cuda.Device().synchronize()

        if short_scan and len(geoms) > 1:
            raise NotImplementedError("Parker weighting not implemented")

        self._filter(projections)