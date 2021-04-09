import matplotlib.pyplot as plt
import numpy as np

from astrapy import bp, fp
from astrapy import Detector, Geometry, rotate

# geometry spinning
geom_t0 = Geometry(
    [-10, 0., 0.],
    [20, 0., 0.],
    [0, 1, 0],
    [0, 0, 1],
    Detector(99, 141, .01, .01))

angles = np.linspace(0, 8 * np.pi, 719)
geoms = [rotate(geom_t0, roll=a, yaw=a) for a in angles]
vol_min, vol_max = [-.2] * 3, [.2] * 3

# cube with random voxels
vol = np.zeros([100] * 3)
vol[25:75, 25:75, 25:75] = np.random.random([50] * 3)
vol[35:65, 35:65, 35:65] += np.random.random([30] * 3)
vol[45:55, 45:55, 45:55] += 1.

# forward project
projs = fp(vol, geoms, vol_min, vol_max)
plt.figure()
for p in projs[0:100:2]:
    plt.cla()
    plt.imshow(p)
    plt.pause(.1)
plt.close()

# backproject
vol2 = bp(projs, geoms, vol.shape, vol_min, vol_max)
plt.figure()
for sl in range(0, vol2.shape[-1]):
    plt.cla()
    plt.imshow(vol2[..., sl], vmin=0, vmax=.02)
    plt.pause(.1)
plt.close()
