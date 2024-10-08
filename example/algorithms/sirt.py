"""Demonstrates a SIRT algorithm with callback function."""
import kernelkit as kk
import matplotlib.pyplot as plt
import numpy as np


# Projection geometries of a cube in the 3D space
geom_t0 = kk.ProjectionGeometry(
    source_position=[-10.0, 0.0, 0.0],
    detector_position=[20.0, 0.0, 0.0],
    detector=kk.Detector(rows=150, cols=200, pixel_height=0.01, pixel_width=0.01),
)
angles = np.linspace(0, 2 * np.pi, 250, False)
proj_geoms = [kk.rotate(geom_t0, yaw=a) for a in angles]

# volume and volume geometry
gt = np.zeros((100, 100, 100))
gt[25:75, 25:75, 25:75] = np.ones([50] * 3)
gt[35:65, 35:65, 35:65] += np.ones([30] * 3)
gt[45:55, 45:55, 45:55] += 1.0
vol_geom = kk.resolve_volume_geometry(
    shape=gt.shape, extent_min=[-0.2] * 3, extent_max=[0.2] * 3
)

# simulate some projections
projs = kk.fp(gt, proj_geoms, vol_geom)

# set up a plotting callback function
fig, axs = plt.subplots(1, 2)


def callback(i, x, y):
    """Callback function that plots residual projections and the volume."""
    j = (i * 10) % len(proj_geoms)
    k = 25 + ((i * 5) % 50)
    y_j = y[j].get()
    x_k = x[..., k].get()
    axs[0].clear()
    axs[0].set_title(f"Iter {i}: residual projection {j}")
    axs[0].imshow(y_j, vmin=-0.25, vmax=0.25)
    axs[1].clear()
    axs[1].set_title(f"Iter {i}: volume slice {k}")
    axs[1].imshow(x_k, vmax=3.0)
    plt.pause(00.01)

# run the Simultaneous Iterative Reconstruction Technique (SIRT)
kk.sirt(projs, proj_geoms, vol_geom, iters=200, callback=callback)
