import matplotlib.pyplot as plt
import numpy as np
import astrapy as ap

# custom geometry, 99x141 detector, 1mm pixel size
# 10cm source-object distance, 20cm object-detector distance along x-axis
geom_t0 = (ap.ProjectionGeometry([-10, 0, 0], [20, 0, 0],
                                 [0, 1, 0], [0, 0, 1],
                                 ap.Detector(99, 141, .01, .01)))
geom_t0 = ap.rotate(geom_t0, yaw=.5 * np.pi, roll=.2 * np.pi, pitch=.1 * np.pi)

# angles from 0 to 2pi, 719 projections
angles = np.linspace(0, 2 * np.pi, 719, False)
geoms = [ap.rotate(geom_t0, yaw=a) for a in angles]
vol_min, vol_max = [-.2, -.2 * 1.1, -.2 * 1.2], [.2, .2 * 1.1, .2 * 1.2]

# cube with random voxels
vol = np.zeros([100, 110, 120])  # x, y, z
vol[25:75, 25:75, 25:75] = np.random.random([50] * 3)
vol[35:65, 35:65, 35:65] += np.random.random([30] * 3)
vol[45:55, 45:55, 45:55] += 1.

# forward project
projs = ap.fp(vol, geoms, vol_min, vol_max)
plt.figure()
for p in projs[0:100:4]:
    plt.cla()
    plt.imshow(p)
    plt.pause(.001)
plt.close()

# backproject
vol2 = ap.fdk(projs, geoms, vol.shape, vol_min, vol_max, verbose=True)
plt.figure()
for sl in range(0, vol2.shape[-2], 4):
    plt.cla()
    plt.imshow(vol2[..., sl], vmin=0, vmax=.02)
    plt.pause(.002)
plt.close()
