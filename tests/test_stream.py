import itertools

import cupy as cp
import numpy as np
from tqdm import tqdm

import kernelkit as kk


def test_stream():
    VOXELS = (100, 10, 10)
    DETECTOR_SHAPE = (100, 200)
    NR_ANGLES = 300
    EXT_MIN = np.array((-.5, -.5, -.5)) / 100 * VOXELS * 60
    EXT_MAX = np.array((.5, .5, .5)) / 100 * VOXELS * 60
    SRC_DIST = 100.
    DET_DIST = 0.
    geom_t0 = kk.ProjectionGeometry([-SRC_DIST, 0, 0.], [DET_DIST, 0., 0.],
                                    [0., 1., 0.], [0., 0., 1.],
                                    kk.Detector(*DETECTOR_SHAPE, 1., 1.))
    geoms = [kk.rotate(geom_t0, yaw=a * 2. * np.pi / NR_ANGLES)
             for a in range(NR_ANGLES)]

    vol_data = cp.zeros(VOXELS, dtype=cp.float32)
    vol_data[17:113, 17:113, 17:113] = 1
    vol_data[33:97, 33:97, 33:97] = 0
    fp = kk.ForwardProjector()
    fp.projection_geometry = geoms
    fp.volume = vol_data
    proj_data = fp(EXT_MIN, EXT_MAX)

    bp = kk.ConeBackprojection()
    params = bp.geoms2params(kk.GeometrySequence.fromList(geoms),
                             vol_data.shape, EXT_MIN, EXT_MAX)
    with cp.cuda.stream.Stream(non_blocking=True):
        for _ in tqdm(itertools.count()):
            txt = kk.copy_to_texture(proj_data)
            vol_data.fill(0.)
            bp(txt, params, vol_data, EXT_MIN, EXT_MAX)
            vol_data[...] = cp.reshape(vol_data, tuple(reversed(vol_data.shape))).T
    bp2 = kk.BackProjector()
    bp2.projection_geometry = geoms
    bp2.projections = proj_data
    f2 = bp2(EXT_MIN, EXT_MAX, volume_shape=VOXELS)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3)
    sl = VOXELS[-1] // 2
    axs[0].cla()
    axs[0].imshow(vol_data[..., sl].get())
    axs[0].set_title("With stream")
    axs[1].cla()
    axs[1].imshow(f2[..., sl].get())
    axs[1].set_title("Without stream")
    axs[2].cla()
    axs[2].imshow(np.abs(vol_data[..., sl].get() - f2[..., sl].get()))
    axs[2].set_title("Difference")
    plt.pause(.001)
    plt.show()

    cp.testing.assert_allclose(f, f2, atol=0., rtol=0.001)


def test_graph():
    VOXELS = (100, 110, 120)
    DETECTOR_SHAPE = (100, 200)
    NR_ANGLES = 300
    EXT_MIN = np.array((-.5, -.5, -.5)) / 100 * VOXELS * 60
    EXT_MAX = np.array((.5, .5, .5)) / 100 * VOXELS * 60
    SRC_DIST = 100.
    DET_DIST = 0.
    geom_t0 = kk.ProjectionGeometry([-SRC_DIST, 0, 0.], [DET_DIST, 0., 0.],
                                    [0., 1., 0.], [0., 0., 1.],
                                    kk.Detector(*DETECTOR_SHAPE, 1., 1.))
    geoms = [kk.rotate(geom_t0, yaw=a * 2. * np.pi / NR_ANGLES)
             for a in range(NR_ANGLES)]

    vol_data = cp.zeros(VOXELS, dtype=cp.float32)
    vol_data[17:113, 17:113, 17:113] = 1
    vol_data[33:97, 33:97, 33:97] = 0
    vol_geom = kk.resolve_volume_geometry(
        shape=vol_data.shape,
       extent_min=EXT_MIN, extent_max=EXT_MAX)

    fp = kk.ForwardProjector()
    fp.projection_geometry = geoms
    fp.volume = vol_data
    fp.volume_geometry = vol_geom
    fp.projections = cp.zeros((NR_ANGLES, *DETECTOR_SHAPE), dtype=cp.float32)
    fp()

    bp = kk.kernels.VoxelDrivenConeBP()
    params = bp.geoms2params(kk.GeometrySequence.fromList(geoms), vol_geom)
    txt = kk.kernel.copy_to_texture(fp.projections)

    with cp.cuda.stream.Stream() as stream:
        vol_data.fill(0.)
        stream.begin_capture()
        bp(txt, params, vol_data, vol_geom)
        graph = stream.end_capture()
        graph.launch(stream)
        vol_data_stored = cp.copy(vol_data)

    with cp.cuda.stream.Stream() as stream:
        vol_data.fill(0.)
        graph.launch(stream)
        stream.synchronize()

    cp.testing.assert_allclose(vol_data_stored, vol_data, atol=0., rtol=0.001)
