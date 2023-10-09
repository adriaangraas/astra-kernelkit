"""Demonstrates the Voxel-Driven Cone Beam Backprojection."""
import numpy as np
import cupy as cp
import kernelkit as kk
import matplotlib.pyplot as plt

geom_t0 = kk.ProjectionGeometry(
    source_position=[-10.0, 0.0, 0.0],
    detector_position=[20.0, 0.0, 0.0],
    detector=kk.Detector(rows=150, cols=200, pixel_height=0.01, pixel_width=0.01),
)
angles = np.linspace(0, 2 * np.pi, 500, False)
proj_geoms = [kk.rotate(geom_t0, yaw=a) for a in angles]

# cube with random voxels
vol_geom = kk.resolve_volume_geometry(
    shape=[100] * 3, extent_min=[-0.2] * 3, extent_max=[0.2] * 3)
vol = np.zeros(vol_geom.shape)
vol[25:75, 25:75, 25:75] = 1.0

# create projection data
projs = kk.fp(vol, proj_geoms, vol_geom)

# output, note dtype=cp.float32 for GPU
out = cp.zeros_like(vol, dtype=cp.float32)

# set up a kernel with some custom kernel parameters
K = kk.kernels.VoxelDrivenConeBP(
    voxels_per_block=(100, 2, 100),  # threads per block must be <= 1024
    projs_per_block=500,
)

# compile the kernel
K.compile(
    max_projs=1024,  # kernel is compiled for 1024 geometries
    texture=K.TextureFetching.Tex3D)  # and for tex3D() commands

# Kernel parameters are precomputed values that the projector needs,
# for the voxel-driven cone beam backprojection these are based
# on the projection and volume geometries.
params = K.geoms2params(proj_geoms, vol_geom)

# Uploads the parameters to constant memory
K.set_params(params)

# Convert the projections to CUDA array objects
txts = kk.kernel.copy_to_texture(projs)

# Launch and execute the kernels
K(txts, out, vol_geom)

# The voxel-driven kernel writes in (z, y, x) order, so we need to
# transpose the output to (x, y, z) order. Since the transpose is not
# without cost, it must be manually indicated. Projectors will return a view.
out[...] = cp.reshape(out, tuple(reversed(out.shape))).T
out = out.get()

# it's normal to see a blurry cube, no FDK filter has been applied
plt.figure()
for sl in range(0, out.shape[-2], 4):
    plt.cla()
    plt.imshow(out[..., sl], vmin=0, vmax=1.00)
    plt.pause(0.115)
plt.close()
