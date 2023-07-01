import itertools

import cupy as cp
import numpy as np
import pytest

import astrapy.kernels
import astrapy as ap
from astrapy.kernel import copy_to_texture


@pytest.fixture
def fpkern():
    return astrapy.kernels.ConeProjection()


@pytest.fixture
def bpkern():
    return astrapy.kernels.ConeBackprojection()


@pytest.fixture()
def proj():
    rows, cols = 9, 9
    return cp.zeros((rows, cols), dtype=cp.float32)


def volume(s1=10, s2=None, s3=None):
    if s2 is None:
        s2 = s1
    if s3 is None:
        s3 = s2
    vol_shp = (s1, s2, s3)
    vol = cp.zeros(vol_shp, cp.float32)
    return vol, vol_shp


def geom(rows=9,
         cols=9,
         src_dist=5.,
         det_dist=5.,
         px_h=1.,
         px_w=1.):
    return ap.ProjectionGeometry(
        [-src_dist, 0, 0],
        [det_dist, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        ap.Detector(rows, cols, px_h, px_w))


@pytest.mark.parametrize('vol_fill, vol_ext, vol_sz',
    itertools.product((-1, 0, 1, 13e7), (2 / 3, 1., 2), (10, 13, 21)))
def test_scaling_ray_integral_through_cube(
    fpkern,
    proj,
    vol_fill,
    vol_ext,
    vol_sz
):
    vol, vol_shp = volume(vol_sz)
    vol.fill(vol_fill)
    vol_txt = copy_to_texture(vol)
    vol_geom = ap.resolve_volume_geometry(
        shape=vol_shp,
        extent_min=np.array((-.5, -.5, -.5)) * vol_ext,
        extent_max=np.array((.5, .5, .5)) * vol_ext)
    fpkern(
        volume_texture=vol_txt,
        volume_geometry=vol_geom,
        projection_geometry=[geom()],
        projections=[proj])
    proj_cpu = proj.get()
    assert proj_cpu.shape == proj.shape
    assert proj_cpu[4, 4] == pytest.approx(vol_fill * vol_ext, 1e-6)

