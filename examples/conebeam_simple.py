import matplotlib.pyplot as plt
import numpy as np

from astrapy import bp, fp
from astrapy.geom3d import Detector, Geometry, rotate


def simple_geom(rows, cols, tube_dist, det_dist, px_w=.01, px_h=.01):
    """Make a simple geometry"""

    return Geometry(
        [-tube_dist, 0., 0.],
        [det_dist, 0., 0.],
        [0, 1, 0],
        [0, 0, 1],
        Detector(rows, cols, px_w, px_h))


# geometry spinning
geom_t0 = simple_geom(tube_dist=10., det_dist=20., rows=99, cols=141)
angles = np.linspace(0, 8 * np.pi, 720)
geoms = [rotate(geom_t0, roll=a, yaw=a) for a in angles]
vol_min, vol_max = [-.2] * 3, [.2] * 3

# cube with random voxels
vol = np.zeros((100, 100, 100))
vol[25:75, 25:75, 25:75] = np.random.random([50] * 3)

# forward project
projs = fp(vol, geoms, vol_min, vol_max)
plt.figure()
for p in projs[0:100:4]:
    plt.cla()
    plt.imshow(p)
    plt.pause(.1)

# backproject
vol2 = bp(projs, geoms, vol.shape, vol_min, vol_max)
plt.figure()
for sl in range(0, vol2.shape[2]):
    plt.cla()
    plt.imshow(vol2[..., sl], vmin=0, vmax=1e-2)
    plt.pause(.001)
