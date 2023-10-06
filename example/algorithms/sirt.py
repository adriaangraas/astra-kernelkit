import kernelkit as kk
import matplotlib.pyplot as plt
import numpy as np


def geometries(n_angles=250):
    """Projection geometries of a cube in the 3D space."""
    geom_t0 = kk.ProjectionGeometry(
        source_position=[-10.0, 0.0, 0.0],
        detector_position=[20.0, 0.0, 0.0],
        detector=kk.Detector(rows=150, cols=200, pixel_height=0.01, pixel_width=0.01),
    )
    angles = np.linspace(0, 2 * np.pi, n_angles, False)
    return [kk.rotate(geom_t0, yaw=a) for a in angles]


if __name__ == "__main__":
    proj_geoms = geometries()
    vol_geom = kk.resolve_volume_geometry(
        shape=[100] * 3, extent_min=[-0.2] * 3, extent_max=[0.2] * 3
    )

    # simulate some projections
    gt = np.zeros(vol_geom.shape)
    gt[25:75, 25:75, 25:75] = np.random.random([50] * 3)
    gt[35:65, 35:65, 35:65] += np.random.random([30] * 3)
    gt[45:55, 45:55, 45:55] += 1.0
    projs = kk.fp(gt, proj_geoms, vol_geom)

    fig, axs = plt.subplots(1, 2)
    def callback(i, x, y):
        """Callback function that plots residual projections and the volume."""
        j = (i * 10) % len(proj_geoms)
        axs[0].clear()
        axs[0].set_title(f"Iter {i}: residual projection {j}")
        axs[0].imshow(y[j].get(), vmin=-0.25, vmax=0.25)
        axs[1].clear()
        k = 25 + ((i * 5) % 50)
        axs[1].set_title(f"Iter {i}: volume slice {k}")
        axs[1].imshow(x[..., k].get(), vmax=3.0)
        plt.pause(00.01)

    # run the Simultaneous Iterative Reconstruction Technique (SIRT)
    _ = kk.sirt(projs, proj_geoms, vol_geom, iters=2000, callback=callback)
    plt.show()
