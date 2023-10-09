"""Demonstrates how to use a CUDA graph with CuPy using an FDK algorithm."""
import cupy as cp
import kernelkit as kk
import matplotlib.pyplot as plt
import numpy as np

# projection geometries of a cube in the 3D space
geom_t0 = kk.ProjectionGeometry(
    source_position=[-10.0, 0.0, 0.0],
    detector_position=[20.0, 0.0, 0.0],
    detector=kk.Detector(rows=150, cols=200, pixel_height=0.01,
                         pixel_width=0.01),
)
angles = np.linspace(0, 2 * np.pi, 1000, False)
proj_geoms = [kk.rotate(geom_t0, yaw=a) for a in angles]

# volume and volume geometries
vol = np.zeros((100, 110, 120), dtype=np.float32)  # a cube with random voxels
vol[25:45, 25:45, 25:45] = np.random.random([20] * 3)
vol[35:65, 35:65, 35:65] += np.random.random([30] * 3)
vol_geom = kk.resolve_volume_geometry(
    vol.shape, [-0.2, None, None], [0.2, None, None], verbose=False)

# simulate some projection data
projs = cp.zeros((len(proj_geoms), *proj_geoms[0].detector.shape),
                 dtype=cp.float32)
kk.fp(vol, proj_geoms, vol_geom, out=projs)
out = cp.zeros_like(vol)

# backproject
stream = cp.cuda.Stream()
kk.processing.preweight(projs, proj_geoms)
kk.processing.filter(projs, filter='ramlak')
with stream:
    # Warm-up required, as it creates geometry and texture first.
    # Without warm-up the memory is not yet allocated, and memory
    # allocation is not part of any graph (we don't use stream-ordered
    # memory allocation yet).
    ptor = kk.BackProjector(texture_type='array')
    ptor.projection_geometry = proj_geoms
    ptor.volume_geometry = vol_geom
    ptor.projections = projs
    ptor.volume = out
    ptor()

    # Repeat, now with all the memory in place. Note that geometries
    # will be fixed in the graph.
    stream.begin_capture()
    ptor.projections = projs  # include a texture transfer in the graph
    ptor()
    graph = stream.end_capture()

for offset in (35, 45, 55):
    # We've got a graph. Let's test it by setting some new projections
    # in the place of the old ones.
    new_vol = cp.zeros_like(vol)
    new_vol[offset:offset + 30, offset:offset + 30, offset:offset + 30] = 1.
    kk.fp(new_vol, proj_geoms, vol_geom, out=projs)
    kk.processing.preweight(projs, proj_geoms)
    kk.processing.filter(projs, filter='ramlak')

    # reset volume
    out.fill(0.)
    graph.launch()

    # output should be the reconstructed new volume
    plt.figure()
    for sl in range(0, out.shape[-2], 4):
        plt.cla()
        plt.imshow(out[..., sl].get(), vmin=0, vmax=0.01)
        plt.pause(.002)
    plt.close()