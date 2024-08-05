import kernelkit as kk
import cupy as cp
import numpy as np
import pytest

ROWS, COLS = (100, 200)
VOL_SHAPE = (100, 110, 120)
NR_ANGLES = 100


@pytest.fixture()
def projs():
    return cp.zeros((NR_ANGLES, ROWS, COLS), dtype=cp.float32)


@pytest.fixture()
def vol():
    vol_shp = VOL_SHAPE
    vol = cp.ones(vol_shp, dtype=cp.float32)
    vol[17:113, 17:113, 17:113] = 1
    vol[33:97, 33:97, 33:97] = 0
    return vol


@pytest.fixture()
def proj_geom():
    SRC_DIST = 100.
    DET_DIST = SRC_DIST
    geom_t0 = kk.ProjectionGeometry(
        [-SRC_DIST, 0, 0.],
        [DET_DIST, 0., 0.],
        kk.Detector(rows=ROWS, cols=COLS,
                    pixel_width=3./COLS, pixel_height=3./ROWS)
    )

    geoms = [
        kk.rotate(
            geom_t0,
            yaw=a * 2. * np.pi / NR_ANGLES)
            for a in range(NR_ANGLES)
    ]
    return geoms


@pytest.fixture()
def vol_geom():
    return kk.resolve_volume_geometry(
        shape=VOL_SHAPE,
        extent_min=[-.5, None, None],
        extent_max=[.5, None, None],
    )


def test_op(projs, vol, proj_geom, vol_geom):
    """Test if KernelKit and Toolbox outcomes correspond."""
    volT = cp.ascontiguousarray(cp.transpose(vol, (2, 1, 0)))

    A_kk = kk.XrayTransform(
        proj_geom,
        vol_geom,
        projection_axes=(1, 0, 2),
        volume_axes=(2, 1, 0),
        bp_kwargs={'texture_type': 'array'}
    )

    A_tb = kk.XrayTransform(
        proj_geom,
        vol_geom,
        use_toolbox=True,
        projection_axes=(1, 0, 2),
        volume_axes=(2, 1, 0))

    out_kk = A_kk(volT)
    out_tb = A_tb(volT)

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # for i in range(NR_ANGLES):
        # plt.cla()
        # axs[0].set_title("KernelKit")
        # axs[0].imshow(out_kk[:, i].get())
        # axs[1].set_title("Toolbox")
        # axs[1].imshow(out_tb[:, i].get())
        # axs[2].set_title("Diff")
        # axs[2].imshow((out_tb[:, i] - out_kk[:, i]).get(), vmin=-1e-5, vmax=1e-5)
        # plt.pause(1.)

    for i in range(NR_ANGLES):
        cp.testing.assert_array_almost_equal(
            out_kk[:, i],
            out_tb[:, i],
            decimal=3, verbose=True,
            err_msg=f"At angle {i}")

    x_kk = A_kk.T(out_kk)
    x_tb = A_tb.T(out_tb)

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # for i in range(A_kk.T.domain_shape[0]):
    #     plt.cla()
    #     axs[0].set_title("KernelKit")
    #     axs[0].imshow(x_kk[i].get())
    #     axs[1].set_title("Toolbox")
    #     axs[1].imshow(x_tb[i].get())
    #     axs[2].set_title("Diff")
    #     axs[2].imshow((x_kk[i] - x_tb[i]).get())#, vmin=-1e-5, vmax=1e-5)
    #     plt.pause(1.)

    for i in range(A_kk.T.domain_shape[0]):
        cp.testing.assert_array_almost_equal(
            x_kk[i],
            x_tb[i],
            decimal=4, verbose=True,
            err_msg=f"At slice {i}")
