import itertools

import astrapy as ap
import cupy as cp
import numpy as np
import pytest

DETECTOR_SHAPE = (100, 200)
VOL_SHAPE = (100, 100, 100)
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
        extent_min=(-30.0, None, None),
        extent_max=(30.0, None, None))


def test_forward_ptor(projs, vol, proj_geom, vol_geom):
    ptor = ap.ConeProjector()
    ptor.projections = projs
    ptor.projection_geometry = proj_geom
    ptor.volume_geometry = vol_geom
    ptor.volume = vol
    out = ptor()
    assert out.shape == projs.shape


def test_backward_ptor(projs, vol, proj_geom, vol_geom):
    ptor = ap.ConeBackprojector()
    ptor.projections = projs
    ptor.volume = vol
    ptor.projection_geometry = proj_geom
    ptor.volume_geometry = vol_geom
    out = ptor()
    assert out.shape == vol.shape


def test_astra_compat_projectors(projs, vol, proj_geom, vol_geom):
    fp = ap.AstraCompatConeProjector()
    fp.volume = cp.ascontiguousarray(cp.copy(vol).transpose((2, 1, 0)))
    fp.projections = cp.ascontiguousarray(cp.copy(projs).transpose((1, 0, 2)))
    fp.volume_geometry = vol_geom
    fp.projection_geometry = proj_geom
    fp()
    g = fp.projections.transpose((1, 0, 2))

    fp2 = ap.ConeProjector()
    fp2.volume_geometry = vol_geom
    fp2.projection_geometry = proj_geom
    fp2.volume = vol
    fp2.projections = projs
    fp2()
    g2 = fp2.projections

    # plot results
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2)
    for theta in range(0, g.shape[0], 20):
        axs[0,0].cla()
        axs[0,0].set_title("ASTRA Toolbox")
        axs[0,0].imshow(g[theta].get(), vmin=0., vmax=50.)
        axs[0,1].cla()
        axs[0,1].set_title("This")
        axs[0,1].imshow(g2[theta].get(), vmin=0., vmax=50.)
        axs[1,0].cla()
        axs[1,0].set_title("Difference")
        axs[1,0].imshow(g[theta].get() - g2[theta].get(),
                        vmin=-0.001, vmax=0.001)
        plt.pause(.01)
    # cp.testing.assert_allclose(g, g2, atol=0.01, rtol=0.05)

    bp = ap.AstraCompatConeBackprojector()
    bp.projection_geometry = proj_geom
    bp.volume_geometry = vol_geom
    bp.volume = cp.ascontiguousarray(cp.copy(vol).transpose((2, 1, 0)))
    bp.projections = cp.ascontiguousarray(cp.copy(g).transpose((1, 0, 2)))
    f = bp().transpose((2, 1, 0))

    bp2 = ap.ConeBackprojector()
    bp2.projection_geometry = proj_geom
    bp2.volume_geometry = vol_geom
    bp2.projections = g  # using same data as compat
    bp2.volume = vol
    f2 = bp2()

    # plot results
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3)
    for sl in range(0, f.shape[2], 10):
        axs[0].cla()
        axs[0].imshow(f[..., sl].get())
        axs[0].set_title("ASTRA Toolbox")
        axs[1].cla()
        axs[1].imshow(f2[..., sl].get())
        axs[1].set_title("This")
        axs[2].cla()
        axs[2].imshow(np.abs(f[..., sl].get() - f2[..., sl].get()))
        axs[2].set_title("Difference")
        plt.pause(.001)

    cp.testing.assert_allclose(f, f2, atol=0., rtol=0.05)


@pytest.mark.parametrize(
    'vol_fill, vol_ext, vol_sz',
    itertools.product((None, -1, 1, 13e7), (2 / 3, 1., .2), (10, 13, 21)))
def test_matching_singleproj(fpkern, bpkern, vol_fill, vol_ext, vol_sz):
    rows, cols = 99, 144
    geoms = geom(src_dist=1., det_dist=20.,
                 rows=rows, cols=cols,
                 px_h=0.50, px_w=.50)
    vol_shp = (100, 100, 100)
    ext = np.array((.5, .5, .5)) * vol_ext

    if vol_fill is not None:
        f = cp.zeros(vol_shp, dtype=cp.float32)
        f.fill(vol_fill)
    else:
        f = cp.random.random(vol_shp, dtype=cp.float32)

    fp = ap.ConeProjector()
    fp.volume = f
    fp.projection_geometry = [geoms]
    g = fp(-ext, ext)

    bp = ap.ConeBackprojector()
    bp.projections = g
    bp.projection_geometry = [geoms]
    f2 = bp(-ext, ext, volume_shape=f.shape)

    # <Ax, Ax> == <A^* A x, x>
    AxAx = cp.inner(g.flatten(), g.flatten()).get()
    xAtAx = cp.inner(f2.flatten(), f.flatten()).get()
    assert AxAx == pytest.approx(xAtAx, rel=.05)