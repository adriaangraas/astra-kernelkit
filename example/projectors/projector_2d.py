"""Demonstrates how to use a CUDA graph with CuPy using an FDK algorithm."""
import cupy as cp
import matplotlib.pyplot as plt

import kernelkit as kk
from kernelkit.projector2d import BackProjector2D
import numpy as np


def geometries(n_angles=719):
    # custom geometry, 99x141 detector, 1mm pixel size
    # 10cm source-object distance, 20cm object-detector distance along x-axis
    geom_t0 = kk.ProjectionGeometry(
        source_position=[-100., 0., 0.],
        detector_position=[100., 0., 0.],
        detector=kk.Detector(
            rows=1,
            cols=1000,
            pixel_height=.001,
            pixel_width=.001))
    angles = np.linspace(0, 2*np.pi, n_angles, False)
    return [kk.rotate(geom_t0, yaw=a) for a in angles]


if __name__ == '__main__':
    proj_geoms = geometries()
    vol = np.zeros((120, 110, 1), np.float32)  # a cube with random voxels
    vol_geom = kk.resolve_volume_geometry(
        vol.shape, [-0.2, None, None], [0.2, None, None], verbose=True)

    projs = cp.ones((len(proj_geoms), *proj_geoms[0].detector.shape), dtype=cp.float32)
    out = cp.zeros_like(vol, dtype=cp.float32)

    ptor = BackProjector2D()
    ptor.projection_geometry = proj_geoms
    ptor.volume_geometry = vol_geom
    ptor.projections = projs
    ptor.volume = out
    ptor()

    plt.figure()
    plt.imshow(out[:, :, 0].get())
    plt.show()
