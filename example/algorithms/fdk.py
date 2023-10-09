"""Demonstrates a Feldkamp-Davis-Kress algorithm."""
import kernelkit as kk
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import cupyx as cpx

# projection geometries of a cube in the 3D space
geom_t0 = kk.ProjectionGeometry(
    source_position=[-10.0, 0.0, 0.0],
    detector_position=[20.0, 0.0, 0.0],
    detector=kk.Detector(rows=150, cols=200, pixel_height=0.01, pixel_width=0.01),
)
angles = np.linspace(0, 2 * np.pi, 1000, False)
proj_geoms = [kk.rotate(geom_t0, yaw=a) for a in angles]

# volume and volume geometries
vol = cpx.zeros_pinned((100, 110, 120), dtype=cp.float32)
vol[25:45, 25:45, 25:45] = np.random.random([20] * 3)
vol[35:65, 35:65, 35:65] += np.random.random([30] * 3)
vol_geom = kk.resolve_volume_geometry(
    vol.shape, [-0.2, None, None], [0.2, None, None], verbose=False
)

# generate projections by forward projection
projs = kk.fp(vol, proj_geoms, vol_geom)

plt.figure()
for p in projs[::16]:
    plt.cla()
    plt.title("Forward projection")
    plt.imshow(p)
    plt.pause(0.001)
plt.close()

# backproject with the Feldkamp-Davis-Kress algorithm
vol2 = kk.fdk(projs, proj_geoms, vol_geom)

plt.figure()
for sl in range(0, vol2.shape[-2], 2):
    plt.title("Feldkamp-Davis-Kress")
    plt.cla()
    plt.imshow(vol2[..., sl], vmin=0, vmax=0.05)
    plt.pause(0.001)
plt.close()
