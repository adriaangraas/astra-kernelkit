import matplotlib.pyplot as plt
import numpy as np
import astrapy as ap

# geometry spinning
geom_t0 = ap.ProjectionGeometry(
    [-10, 0., 0.],
    [10, 0., 0.],
    [0, 1, 0],
    [0, 0, 1],
    ap.Detector(99, 141, .01, .01))
geom_t0 = ap.rotate(geom_t0, yaw=.5 * np.pi, roll=.2 * np.pi, pitch=.1 * np.pi)
angles = np.linspace(0, 2 * np.pi, 300, False)
geoms = [ap.rotate(geom_t0, yaw=a) for a in angles]
vol_geom = ap.resolve_volume_geometry(
    shape=[150] * 3,
    extent_min=[-.2] * 3,
    extent_max=[.2] * 3)

# cube with random voxels
vol = np.zeros(vol_geom.shape)
vol[25:75, 25:75, 25:75] = np.random.random([50] * 3)
vol[35:65, 35:65, 35:65] += np.random.random([30] * 3)
vol[45:55, 45:55, 45:55] += 1.
vol = vol.transpose((2, 1, 0))

# forward project
projs = ap.fp(vol, geoms, vol_geom)

def callbackf(i, x, y):
    if i % 100 == 0:
        plt.figure(1)
        for p in y[::16]:
            plt.cla()
            plt.title(f'Projections at iteration {i}')
            plt.imshow(p.get(), vmin=-.25, vmax=.25)
            plt.pause(00.01)

        plt.figure(2)
        for sl in range(0, x.shape[2], 4):
            plt.cla()
            plt.title(f'Volume at iteration {i}')
            plt.imshow(x[..., sl].get(), vmax=3.0)
            plt.pause(00.01)


vol2 = ap.sirt(
    projs,
    geoms,
    vol_geom,
    iters=2000,
    # callback=callbackf
)

plt.figure(2)
for sl in range(25, vol2.shape[-1]):
    plt.cla()
    plt.title(f'SIRT reconstruction')
    plt.imshow(vol2[..., sl], vmin=0, vmax=3.0)
    plt.pause(.1)
plt.close()
