import astrapy as ap
import cupy as cp
import numpy as np
import pytest

from astrapy import resolve_volume_geometry

DETECTOR_SHAPE = (100, 200)
VOL_SHAPE = (100, 110, 120)
NR_ANGLES = 300


@pytest.fixture()
def projs():
    return cp.zeros((NR_ANGLES, *DETECTOR_SHAPE), dtype=cp.float32)


@pytest.fixture()
def vol():
    vol_shp = VOL_SHAPE
    vol = cp.zeros(vol_shp, dtype=cp.float32)
    vol[17:113, 17:113, 17:113] = 1
    vol[33:97, 33:97, 33:97] = 0
    return vol


@pytest.fixture()
def proj_geom():
    SRC_DIST = 100.
    DET_DIST = 0.
    geom_t0 = ap.ProjectionGeometry([-SRC_DIST, 0, 0.], [DET_DIST, 0., 0.],
                                    [0., 1., 0.], [0., 0., 1.],
                                    ap.Detector(*DETECTOR_SHAPE, 1., 1.))
    geoms = [ap.rotate(geom_t0, yaw=a * 2. * np.pi / NR_ANGLES)
             for a in range(NR_ANGLES)]
    return geoms


@pytest.fixture()
def vol_geom():
    return ap.resolve_volume_geometry(
        shape=VOL_SHAPE,
        extent_min=(-.5 * 60, None, None),
        extent_max=(.5 * 60, None, None))


def test_op(projs, vol, proj_geom, vol_geom, vol_geom_astra):
    op = ap.ConebeamTransform(proj_geom, vol_geom)
    out = op.T(op(vol))

    op2 = ap.ConebeamTransform(proj_geom, vol_geom,
                               astra_compat=True,
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
