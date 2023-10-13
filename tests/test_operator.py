import kernelkit as kk
import cupy as cp
import numpy as np
import pytest

from kernelkit import resolve_volume_geometry

_DET_SHAPE = (100, 200)
_VOL_SHAPE = (100, 110, 120)
_NR_ANGLES = 300


@pytest.fixture()
def projs():
    return cp.zeros((_NR_ANGLES, *_DET_SHAPE), dtype=cp.float32)


@pytest.fixture()
def vol():
    vol_shp = _VOL_SHAPE
    vol = cp.zeros(vol_shp, dtype=cp.float32)
    vol[17:113, 17:113, 17:113] = 1
    vol[33:97, 33:97, 33:97] = 0
    return vol


@pytest.fixture()
def proj_geom():
    SRC_DIST = 100.
    DET_DIST = 0.
    geom_t0 = kk.ProjectionGeometry([-SRC_DIST, 0, 0.], [DET_DIST, 0., 0.],
                                    [0., 1., 0.], [0., 0., 1.],
                                    kk.Detector(*_DET_SHAPE, 1., 1.))
    geoms = [kk.rotate(geom_t0, yaw=a * 2. * np.pi / _NR_ANGLES)
             for a in range(_NR_ANGLES)]
    return geoms


@pytest.fixture()
def vol_geom():
    return kk.resolve_volume_geometry(
        shape=_VOL_SHAPE,
        extent_min=(-.5 * 60, None, None),
        extent_max=(.5 * 60, None, None))


def test_op(projs, vol, proj_geom, vol_geom):
    op = kk.XrayTransform(proj_geom, vol_geom)
    out = op.T(op(vol))

    op2 = kk.XrayTransform(proj_geom, vol_geom,
                           projection_axes=(1, 0, 2),
                           volume_axes=(2, 1, 0))
    vol = cp.ascontiguousarray(vol.transpose(2, 1, 0))
    out2 = op2.T(op2(vol))

    import matplotlib.pyplot as plt
    plt.subplots(1, 2)
    plt.subplot(1, 2, 1)
    plt.imshow(out[0].get())
    plt.subplot(1, 2, 2)
    plt.imshow(out2[0].get())
    plt.show()

    assert cp.allclose(out, out2)
