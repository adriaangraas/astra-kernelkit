"""Demonstrates a simple conebeam forward and back projection."""
import matplotlib.pyplot as plt
import numpy as np
import kernelkit as kk


def geometries(n_angles=719):
    # custom geometry, 99x141 detector, 1mm pixel size
    # 10cm source-object distance, 20cm object-detector distance along x-axis
    geom_t0 = kk.ProjectionGeometry(
        source_position=[-10., 0., 0.],
        detector_position=[20., 0., 0.],
        detector=kk.Detector(
            rows=121,
            cols=141,
            pixel_height=.01,
            pixel_width=.01))
    geom_t0 = kk.rotate(geom_t0,
                        yaw=.5 * np.pi, roll=.2 * np.pi, pitch=.1 * np.pi)
    angles = np.linspace(0, 2 * np.pi, n_angles, False)
    return [kk.rotate(geom_t0, yaw=a) for a in angles]


if __name__ == '__main__':
    proj_geoms = geometries()
    vol = np.zeros((100, 110, 120))  # a cube with random voxels
    vol[25:75, 25:75, 25:75] = np.random.random([50] * 3)
    vol[35:65, 35:65, 35:65] += np.random.random([30] * 3)
    vol[45:55, 45:55, 45:55] += 1.
    vol_geom = kk.resolve_volume_geometry(
        vol.shape, [-.2, None, None], [.2, None, None])

    # forward project
    projs = kk.fp(vol, proj_geoms, vol_geom)

    plt.figure()
    for p in projs[0:100:4]:
        plt.cla()
        plt.imshow(p)
        plt.pause(.001)
    plt.close()

    # backproject
    vol2 = kk.fdk(projs, proj_geoms, vol_geom)

    plt.figure()
    for sl in range(0, vol2.shape[-2], 4):
        plt.cla()
        plt.imshow(vol2[..., sl], vmin=0, vmax=.02)
        plt.pause(.002)
    plt.close()
