import astra
import astra.experimental
import pytest

import astrapy as ap
import numpy as np
import cupy as cp


def test_fp():
    VOXELS = (100, 110, 120)  # my convention: x, y, z
    DETECTOR_SHAPE = (100, 200)
    NR_ANGLES = 100
    EXT_MIN = np.array((-.5, -.5, -.5)) / 100 * VOXELS * 60
    EXT_MAX = np.array((.5, .5, .5)) / 100 * VOXELS * 60
    SRC_DIST = 100.
    DET_DIST = 0.

    vol_data = cp.zeros(VOXELS, dtype=cp.float32)
    vol_data[17:113, 17:113, 17:113] = 1
    vol_data[33:97, 33:97, 33:97] = 0
    # vol_data = cp.ones(VOXELS, dtype=cp.float32)
    proj_data = cp.zeros((NR_ANGLES, *DETECTOR_SHAPE), dtype=cp.float32)

    angles = np.linspace(0, 2 * np.pi, NR_ANGLES, False)

    # ASTRA, set-up geometry
    vectors = np.zeros((len(angles), 12))
    for i in range(len(angles)):
        # source
        vectors[i, 0] = np.sin(angles[i]) * SRC_DIST
        vectors[i, 1] = -np.cos(angles[i]) * SRC_DIST
        vectors[i, 2] = 0
        # center of detector
        vectors[i, 3] = np.sin(angles[i]) * -DET_DIST
        vectors[i, 4] = -np.cos(angles[i]) * -DET_DIST
        vectors[i, 5] = 0
        # vector from detector pixel (0,0) to (0,1)
        vectors[i, 6] = np.cos(angles[i])
        vectors[i, 7] = np.sin(angles[i])
        vectors[i, 8] = 0
        # vector from detector pixel (0,0) to (1,0)
        vectors[i, 9] = 0
        vectors[i, 10] = 0
        vectors[i, 11] = 1

    # make ASTRA volume
    vol_data_astra = cp.ascontiguousarray(cp.transpose(vol_data, (2, 0, 1)))
    z, y, x = vol_data_astra.shape
    vol_link = astra.pythonutils.GPULink(vol_data_astra.data.ptr, x, y, z,
                                         x * 4)
    vol_geom = astra.create_vol_geom(*VOXELS,
                                     EXT_MIN[1], EXT_MAX[1],
                                     EXT_MIN[0], EXT_MAX[0],
                                     EXT_MIN[2], EXT_MAX[2])

    # make ASTRA projs
    proj_data_astra = cp.ascontiguousarray(proj_data.transpose(1, 0, 2))
    z, y, x = proj_data_astra.shape
    proj_link = astra.pythonutils.GPULink(proj_data_astra.data.ptr, x, y, z,
                                          x * 4)
    proj_geom = astra.create_proj_geom('cone_vec', *DETECTOR_SHAPE, vectors)

    # make ASTRA projector
    proj_cfg = {'type': 'cuda3d', 'VolumeGeometry': vol_geom,
                'ProjectionGeometry': proj_geom, 'options': {}}
    projector_id = astra.projector3d.create(proj_cfg)

    # do ASTRA forward projection
    astra.experimental.direct_FPBP3D(projector_id, vol_link, proj_link,
                                     1, "FP")

    # make ASTRApy projection kernel
    geom_t0 = ap.Geometry([-SRC_DIST, 0, 0.], [DET_DIST, 0., 0.],
                          [0., 1., 0.], [0., 0., 1.],
                          ap.Detector(*DETECTOR_SHAPE, 1., 1.))
    geoms = [ap.rotate(geom_t0, yaw=a * 2. * np.pi / NR_ANGLES) for a in range(NR_ANGLES)]
    vol_txt = ap.copy_to_texture(vol_data)
    fp = ap.ConeProjection()
    fp(vol_txt, EXT_MIN, EXT_MAX, geoms, proj_data)

    # plot results
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2)
    for theta in range(0, proj_data_astra.shape[1], 3):
        axs[0,0].cla()
        axs[0,0].set_title("ASTRA Toolbox")
        axs[0,0].imshow(proj_data_astra[:, theta, :].get())
        axs[0,1].cla()
        axs[0,1].set_title("This")
        axs[0,1].imshow(proj_data[theta].get())
        axs[1,0].cla()
        axs[1,0].set_title("Difference")
        axs[1,0].imshow(proj_data_astra[:, theta, :].get() - proj_data[theta].get(),
                      vmin=-.001, vmax=.001)
        plt.pause(.01)

    # make sure we use the same data?
    proj_data[...] = proj_data_astra.transpose((1, 0, 2))

    # do ASTRA backprojection
    astra.experimental.direct_FPBP3D(projector_id, vol_link, proj_link,
                                     1, "BP")

    # do ASTRApy backprojection
    proj_txt = ap.copy_to_texture(proj_data)
    bp = ap.ConeBackprojection()
    params = bp.geoms2params(geoms, vol_data.shape, EXT_MIN, EXT_MAX)
    bp.__call__(proj_txt, params, vol_data, EXT_MIN, EXT_MAX)
    vol_data[...] = cp.reshape(vol_data, tuple(reversed(vol_data.shape))).T

    # plot results
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3)
    for sl in range(0, vol_data.shape[2], 8):
        axs[0].cla()
        axs[1].cla()
        axs[2].cla()
        axs[0].imshow(vol_data_astra[sl].get())
        axs[0].set_title("ASTRA Toolbox")
        axs[1].imshow(vol_data[..., sl].get())
        axs[1].set_title("This")
        axs[2].imshow(np.abs(vol_data_astra[sl].get()
                      - vol_data[..., sl].get()),
                      vmin=0.0, vmax=1.1)
        axs[2].set_title("Difference")
        plt.pause(.01)
    plt.show()