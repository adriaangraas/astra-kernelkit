import itertools

import kernelkit as kk
from kernelkit.toolbox_support import ForwardProjectorAdapter, \
    BackprojectorAdapter
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
    geom_t0 = kk.ProjectionGeometry([-SRC_DIST, 0, 0.], [DET_DIST, 0., 0.],
                                    kk.Detector(*DETECTOR_SHAPE, 1., 1.))
    geoms = [kk.rotate(geom_t0, yaw=a * 2. * np.pi / NR_ANGLES)
             for a in range(NR_ANGLES)]
    return geoms


@pytest.fixture()
def vol_geom():
    return kk.resolve_volume_geometry(
        shape=VOL_SHAPE,
        extent_min=(-30.0, None, None),
        extent_max=(30.0, None, None))


def test_forward_ptor(projs, vol, proj_geom, vol_geom):
    ptor = kk.ForwardProjector()
    ptor.projections = projs
    ptor.projection_geometry = proj_geom
    ptor.volume_geometry = vol_geom
    ptor.volume = vol
    out = ptor()
    assert out.shape == projs.shape


def test_backward_ptor(projs, vol, proj_geom, vol_geom):
    ptor = kk.BackProjector()
    ptor.projections = projs
    ptor.volume = vol
    ptor.projection_geometry = proj_geom
    ptor.volume_geometry = vol_geom
    out = ptor()
    assert out.shape == vol.shape


def test_astra_compat_projectors(projs, vol, proj_geom, vol_geom):
    fp_tb = ForwardProjectorAdapter()
    fp_tb.volume = cp.ascontiguousarray(cp.copy(vol).transpose((2, 1, 0)))
    fp_tb.projections = cp.ascontiguousarray(
        cp.copy(projs).transpose((1, 0, 2)))
    fp_tb.volume_geometry = vol_geom
    fp_tb.projection_geometry = proj_geom
    fp_tb()
    g_tb = fp_tb.projections.transpose((1, 0, 2))

    fp_kk = kk.ForwardProjector()
    fp_kk.volume_geometry = vol_geom
    fp_kk.projection_geometry = proj_geom
    fp_kk.volume = vol
    fp_kk.projections = projs
    fp_kk()
    g_kk = fp_kk.projections

    # plot results
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3)
    for theta in range(0, g_tb.shape[0], 20):
        axs[0].cla()
        axs[0].set_title("ASTRA Toolbox")
        axs[0].imshow(g_tb[theta].get(), vmin=0., vmax=50.)
        axs[1].cla()
        axs[1].set_title("ASTRA KernelKit")
        axs[1].imshow(g_kk[theta].get(), vmin=0., vmax=50.)
        axs[2].cla()
        axs[2].set_title("Difference")
        axs[2].imshow(g_tb[theta].get() - g_kk[theta].get(),
                         vmin=-0.001, vmax=0.001)
        plt.pause(.01)
    # cp.testing.assert_allclose(g, g2, atol=0.01, rtol=0.05)

    bp = BackprojectorAdapter()
    bp.projection_geometry = proj_geom
    bp.volume_geometry = vol_geom
    bp.volume = cp.ascontiguousarray(cp.zeros_like(vol).transpose((2, 1, 0)))
    bp.projections = cp.ascontiguousarray(cp.copy(g_tb).transpose((1, 0, 2)))
    bp()
    f = bp.volume.transpose((2, 1, 0))

    bp2 = kk.BackProjector()
    bp2.projection_geometry = proj_geom
    bp2.volume_geometry = vol_geom
    bp2.projections = g_tb  # using same data as compat
    bp2.volume = vol
    bp2()
    f2 = bp2.volume

    # plot results
    fig, axs = plt.subplots(1, 3)
    for sl in range(0, f.shape[2], 4):
        axs[0].cla()
        axs[0].imshow(f[..., sl].get())
        axs[0].set_title("ASTRA Toolbox")
        axs[1].cla()
        axs[1].imshow(f2[..., sl].get())
        axs[1].set_title("ASTRA KernelKit")
        axs[2].cla()
        axs[2].imshow(np.abs(f[..., sl].get() - f2[..., sl].get()))
        axs[2].set_title("Difference")
        plt.pause(.001)

    cp.testing.assert_allclose(f, f2, atol=0., rtol=0.05)


@pytest.mark.xfail
def test_pitched_projectors_dont_copy():
    """Need to test this with SIRT."""
    raise NotImplementedError


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

    fp = kk.ForwardProjector()
    fp.volume = f
    fp.projection_geometry = [geoms]
    g = fp(-ext, ext)

    bp = kk.BackProjector()
    bp.projections = g
    bp.projection_geometry = [geoms]
    f2 = bp(-ext, ext, volume_shape=f.shape)

    # <Ax, Ax> == <A^t A x, x>
    AxAx = cp.inner(g.flatten(), g.flatten()).get()
    xAtAx = cp.inner(f2.flatten(), f.flatten()).get()
    assert AxAx == pytest.approx(xAtAx, rel=.05)