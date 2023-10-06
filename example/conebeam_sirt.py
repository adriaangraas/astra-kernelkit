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
    vol_geom = kk.resolve_volume_geometry(
        shape=[150] * 3,
        extent_min=[-.2] * 3,
        extent_max=[.2] * 3)

    # simulate some projections
    gt = np.zeros(vol_geom.shape)
    gt[25:75, 25:75, 25:75] = np.random.random([50] * 3)
    gt[35:65, 35:65, 35:65] += np.random.random([30] * 3)
    gt[45:55, 45:55, 45:55] += 1.
    gt = gt.transpose((2, 1, 0))
    projs = kk.fp(gt, proj_geoms, vol_geom)

    def callback(i, x, y):
        """Callback function that plots residual projections and the volume."""
        if i % 100 != 0:
            return
        plt.figure(1)
        for p in y[::16]:
            plt.cla()
            plt.title(f'Residual projections at iteration {i}')
            plt.imshow(p.get(), vmin=-.25, vmax=.25)
            plt.pause(00.01)
        plt.figure(2)
        for sl in range(0, x.shape[2], 4):
            plt.cla()
            plt.title(f'Volume at iteration {i}')
            plt.imshow(x[..., sl].get(), vmax=3.0)
            plt.pause(00.01)


    # run the Simultaneous Iterative Reconstruction Technique (SIRT)
    vol2 = kk.sirt(projs, proj_geoms, vol_geom, iters=2000, callback=callback)
    plt.show()