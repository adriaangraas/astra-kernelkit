import matplotlib.pyplot as plt
import numpy as np
import astrapy as ap

# geometry spinning
geom_t0 = ap.Geometry(
    [-10, 0., 0.],
    [20, 0., 0.],
    [0, 1, 0],
    [0, 0, 1],
    ap.Detector(99, 141, .01, .01))

geom_t0 = ap.rotate(geom_t0, yaw=.5*np.pi, roll=.2 * np.pi, pitch=.1 * np.pi)
angles = np.linspace(0, 2 * np.pi, 200, False)
geoms = [ap.rotate(geom_t0, yaw=a) for a in angles]
vol_min, vol_max = [-.2] * 3, [.2] * 3

# cube with random voxels
vol = np.zeros([100] * 3)
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

vol2 = ap.sirt_experimental(projs, geoms, vol.shape, vol_min, vol_max,
                         iters=200)
plt.figure()
for sl in range(25, vol2.shape[-1]):
    plt.cla()
    plt.imshow(vol2[..., sl], vmin=0, vmax=2.)
    plt.pause(.001)
plt.close()