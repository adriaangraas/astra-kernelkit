import warnings

import cupy as cp
import numpy as np
from tqdm import tqdm

from astrapy.geom3d import Geometry


def filter(projections,
           verbose=False):
    xp = cp.get_array_module(projections[0])
    ramlak = xp.linspace(-1, 1, num=projections[0].shape[1] // 2 + 1)
    ramlak = xp.abs(ramlak)
    for p in tqdm(projections, desc="Filtering", disable=not verbose):
        assert cp.get_array_module(p) == xp, (
            "Arrays need to be all cupy or all numpy.")
        f = xp.fft.fftshift(xp.fft.rfft(p))
        f *= ramlak  # complex mult with ramp filter
        p[...] = xp.fft.irfft(xp.fft.ifftshift(f), n=p.shape[1])


def preweight(projections,
              geoms: list[Geometry],
              verbose: bool = False):
    """Pixelwise rescaling to compensate for ray length in conebeam images"""
    xp = cp.get_array_module(projections[0])
    # prepare computation of all pixel vectors
    rows, cols = xp.mgrid[0:geoms[0].detector.rows, 0:geoms[0].detector.cols]
    rows_view = xp.repeat(rows[:, :, xp.newaxis], 3, 2)
    cols_view = xp.repeat(cols[:, :, xp.newaxis], 3, 2)

    for p, g in tqdm(zip(projections, geoms), disable=not verbose,
                     desc="Preweighting"):
        assert cp.get_array_module(p) == xp, (
            "Arrays need to be all cupy or all numpy.")
        assert g.detector.rows == geoms[0].detector.rows
        assert g.detector.cols == geoms[0].detector.cols
        central_ray = np.linalg.norm(g.detector_piercing - g.tube_position)
        pixels = (xp.array(g.detector_extent_min)
                  + cols_view * xp.array(g.u * g.detector.pixel_width)
                  + rows_view * xp.array(g.v * g.detector.pixel_height))
        pixel_rays = xp.linalg.norm(pixels - xp.array(g.tube_position), axis=2)
        p *= xp.divide(central_ray, pixel_rays)