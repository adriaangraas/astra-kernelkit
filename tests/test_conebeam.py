import itertools

import cupy as cp
import numpy as np
import pytest

from astrapy.geom import Detector
from astrapy.kernel import copy_to_texture
from astrapy.kernels import (Geometry, ConeBackprojection,
                             ConeProjection)

@pytest.fixture
def fpkern():
    return ConeProjection()


@pytest.fixture
def bpkern():
    return ConeBackprojection()


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
         tube_dist=5.,
         det_dist=5.,
         px_w=1.,
         px_h=1.):
    return Geometry(
        [-tube_dist, 0, 0],
        [det_dist, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        Detector(rows, cols, px_w, px_h))


@pytest.mark.parametrize(
    'vol_fill, vol_ext, vol_sz',
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
    fpkern(
        volume_texture=vol_txt,
        volume_extent_min=np.array((-.5, -.5, -.5)) * vol_ext,
        volume_extent_max=np.array((.5, .5, .5)) * vol_ext,
        geometries=[geom()],
        projections=[proj])
    proj_cpu = proj.get()
    assert proj_cpu.shape == proj.shape
    assert proj_cpu[4, 4] == pytest.approx(vol_fill * vol_ext, 1e-6)


@pytest.mark.parametrize(
    'vol_fill, vol_ext, vol_sz',
    itertools.product((None, -1, 1, 13e7), (2 / 3, 1., .2), (10, 13, 21)))
def test_matching_singleproj(fpkern, bpkern, vol_fill, vol_ext, vol_sz):
    rows, cols = 99, 144
    geoms = geom(tube_dist=1., det_dist=20.,
                 rows=rows, cols=cols,
                 px_w=0.50, px_h=.50)
    vol_shp = (100, 100, 100)
    ext = np.array((.5, .5, .5)) * vol_ext

    if vol_fill is not None:
        f = cp.zeros(vol_shp, dtype=cp.float32)
        f.fill(vol_fill)
    else:
        f = cp.random.random(vol_shp, dtype=cp.float32)

    g = cp.zeros((rows, cols), dtype=cp.float32)
    fpkern(
        volume_texture=copy_to_texture(f),
        volume_extent_min=-ext,
        volume_extent_max=ext,
        geometries=[geoms],
        projections=[g])

    f2 = cp.zeros_like(f)
    bpkern(
        projections_textures=[copy_to_texture(g)],
        geometries=[geoms],
        volume=f2,
        volume_extent_min=-ext,
        volume_extent_max=ext)

    # <Ax, Ax> == <A^* A x, x>
    AxAx = cp.inner(g.flatten(), g.flatten()).get()
    xAtAx = cp.inner(f2.flatten(), f.flatten()).get()
    assert AxAx == pytest.approx(xAtAx, rel=.05)
