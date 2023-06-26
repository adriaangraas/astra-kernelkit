import astra
import astra.experimental
import pytest

import astrapy as ap
import numpy as np
import cupy as cp

from astrapy.projector import AstraCompatConeBackprojector, \
    AstraCompatConeProjector, ConeBackprojector, ConeProjector


def test_astra_compat_projectors():
    VOXELS = (100, 110, 120)  # my convention: x, y, z
    DETECTOR_SHAPE = (100, 200)
    NR_ANGLES = 300
    EXT_MIN = np.array((-.5, -.5, -.5)) / 100 * VOXELS * 60
    EXT_MAX = np.array((.5, .5, .5)) / 100 * VOXELS * 60

    vol_data = cp.zeros(VOXELS, dtype=cp.float32)
    vol_data[17:113, 17:113, 17:113] = 1
    vol_data[33:97, 33:97, 33:97] = 0
    proj_data = cp.zeros((NR_ANGLES, *DETECTOR_SHAPE), dtype=cp.float32)

    SRC_DIST = 100.
    DET_DIST = 0.
    geom_t0 = ap.Geometry([-SRC_DIST, 0, 0.], [DET_DIST, 0., 0.],
                          [0., 1., 0.], [0., 0., 1.],
                          ap.Detector(*DETECTOR_SHAPE, 1., 1.))
    geoms = [ap.rotate(geom_t0, yaw=a * 2. * np.pi / NR_ANGLES)
             for a in range(NR_ANGLES)]

    fp = AstraCompatConeProjector()
    fp.geometry = geoms
    fp.volume = vol_data
    g = fp(EXT_MIN, EXT_MAX)

    fp2 = ConeProjector()
    fp2.geometry = geoms
    fp2.volume = vol_data
    g2 = fp2(EXT_MIN, EXT_MAX)

    cp.testing.assert_allclose(g, g2, atol=0.01, rtol=0.05)
    # # plot results
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(2, 2)
    # for theta in range(0, g.shape[0], 3):
    #     axs[0,0].cla()
    #     axs[0,0].set_title("ASTRA Toolbox")
    #     axs[0,0].imshow(g[theta].get())
    #     axs[0,1].cla()
    #     axs[0,1].set_title("This")
    #     axs[0,1].imshow(g2[theta].get())
    #     axs[1,0].cla()
    #     axs[1,0].set_title("Difference")
    #     axs[1,0].imshow(g[theta].get() - g2[theta].get(), vmin=-.001, vmax=.001)
    #     plt.pause(.01)

    bp = AstraCompatConeBackprojector()
    bp.geometry = geoms
    bp.projections = g
    f = bp(EXT_MIN, EXT_MAX, volume_shape=VOXELS)

    bp2 = ConeBackprojector()
    bp2.geometry = geoms
    bp2.projections = g  # using same data as compat
    f2 = bp2(EXT_MIN, EXT_MAX, volume_shape=VOXELS)

    cp.testing.assert_allclose(f, f2, atol=0., rtol=0.05)

    # # plot results
    # import matplotlib.pyplot as plt
    #
    # fig, axs = plt.subplots(1, 3)
    # for sl in range(0, f.shape[2], 1):
    #     axs[0].cla()
    #     axs[0].imshow(f[..., sl].get())
    #     axs[0].set_title("ASTRA Toolbox")
    #     axs[1].cla()
    #     axs[1].imshow(f2[..., sl].get())
    #     axs[1].set_title("This")
    #     axs[2].cla()
    #     axs[2].imshow(np.abs(f[..., sl].get() - f2[..., sl].get()))
    #     axs[2].set_title("Difference")
    #     plt.pause(.001)
    # plt.show()