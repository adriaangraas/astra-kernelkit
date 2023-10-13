import numpy as np
import cupy as cp
import kernelkit as kk
import matplotlib.pyplot as plt

NR_ANGELS = 4
assert NR_ANGELS % 2 == 0

# custom geometry, 99x141 detector, 1mm pixel size
# 10cm source-object distance, 20cm object-detector distance along x-axis
geom_t0 = (kk.ProjectionGeometry([-10, 0, 0], [20, 0, 0],
                                 [0, 1, 0], [0, 0, 1],
                                 kk.Detector(99, 141, .01, .01)))
geom_t0 = kk.rotate(geom_t0, yaw=.5 * np.pi, roll=.2 * np.pi, pitch=.1 * np.pi)

# angles from 0 to 2pi, 719 projections
angles = np.linspace(0, 2 * np.pi, NR_ANGELS, False)
proj_geoms = [kk.rotate(geom_t0, yaw=a) for a in angles]
vol_geom = kk.resolve_volume_geometry(
    shape=[100, 110, 120],
    extent_min=[-.2, -.2 * 1.1, -.2 * 1.2],
    extent_max=[.2, .2 * 1.1, .2 * 1.2])

# cube with random voxels
vol = np.zeros(vol_geom.shape)
vol[25:75, 25:75, 25:75] = 1.

# create projection data
projs = np.concatenate(
    (kk.fp(vol, proj_geoms[:1], vol_geom),
     kk.fp(vol, proj_geoms[1:2], vol_geom)),
    axis=0)

out = cp.zeros_like(vol, dtype=cp.float32)

for geoms, data in zip(
    (proj_geoms[:1], proj_geoms[1:2]),
    (projs[:1], projs[1:2])):

    K = kk.kernels.VoxelDrivenConeBP()
    K.compile(1)
    params = K.geoms2params(geoms, vol_geom)
    txts = kk.kernel.copy_to_texture(data)
    K(txts, params, out, vol_geom)

out[...] = cp.reshape(out, tuple(reversed(out.shape))).T
out = out.get()

plt.figure()
for sl in range(0, out.shape[-2], 4):
    plt.cla()
    plt.imshow(out[..., sl])
    plt.pause(.002)
plt.close()