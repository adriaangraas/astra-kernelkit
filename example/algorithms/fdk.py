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
    detector=kk.Detector(rows=150, cols=200,
                         pixel_height=0.01, pixel_width=0.01),
)
angles = np.linspace(0, 2 * np.pi, 256, False)
proj_geoms = [kk.rotate(geom_t0, yaw=a) for a in angles]

# volume and volume geometries
vol = cpx.zeros_pinned((100, 110, 120), dtype=cp.float32)
vol[15:45, 15:45, 15:45] = 1.0
vol[25:65, 25:65, 25:65] += 2.0
vol_geom = kk.resolve_volume_geometry(
    vol.shape, [-0.2, None, None], [0.2, None, None], verbose=False
)

# generate projections by forward projection
projs = kk.fp(vol, proj_geoms, vol_geom)

plt.figure()
for theta, proj in zip(angles[::4], projs[::4]):
    plt.cla()
    plt.xlabel('$u$')
    plt.ylabel('$v$ (parallel to $z$-axis)')
    plt.title(f"Simulated projection at $\\theta={round(theta, 2)}$")
    plt.imshow(proj, cmap='gray', aspect='equal', interpolation='none',
               origin='lower')
    plt.pause(0.001)
plt.close()

# backproject with the Feldkamp-Davis-Kress algorithm
vol2 = kk.fdk(projs, proj_geoms, vol_geom)

plt.figure()
for z in range(10, 40):
    plt.cla()
    plt.title(f"Feldkamp-Davis-Kress $z={z}$")
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.imshow(vol2[..., z].T,
               # vmin=vol.min(), vmax=vol.max(),
               cmap='gray', aspect='equal', interpolation='none',
               origin='lower')
    plt.pause(0.1)
plt.close()
