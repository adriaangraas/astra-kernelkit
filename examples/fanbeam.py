import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

from astrapy import *
from astrapy.kernel import FanProjection, FanBackprojection

# flat detector pixels
nr_pixels = 11
angles = []
points_x = []
points_y = []
points_x2 = []
points_y2 = []
points_x3 = []
points_y3 = []
for i in range(360):
    alpha = i * 2 * np.pi / 360
    det_u = 0.4 * np.array([-np.sin(alpha), np.cos(alpha)])
    tube = -10 * np.array([np.cos(alpha), np.sin(alpha)])
    det = 1 * np.array([np.cos(alpha), np.sin(alpha)])
    angle = Static2DGeometry(*det, *det_u, *tube)
    points_x.append(tube[0])
    points_y.append(tube[1])
    points_x2.append(det[0])
    points_y2.append(det[1])
    points_x3.append(det_u[0])
    points_y3.append(det_u[1])
    angles.append(angle)

# plt.figure()
# plt.scatter(points_x, points_y)
# plt.scatter(points_x2, points_y2)
# plt.scatter(points_x3, points_y3)
# plt.show()

angles = [angles[0]]

# create volume, and projections in RAM
print('Before:\n')
volume_data = np.zeros((10, 10))
volume_data[3,3] = 10
# volume_data[0,0] = 1

volume = Volume(volume_data, (-1, -1), (1, 1))
projection_data = np.ones((len(angles), nr_pixels))
sino = Sinogram(projection_data, (-1,), (1,))

print(sino.data)
print(volume.data)

print('\nPixels after projection:\n')
fp = FanProjection()
sino = fp(volume, sino, angles)

print(volume.data)
# plt.figure()
# plt.imshow(cp.asnumpy(sino.data.T))
# plt.show()

# sino.data[0, 5] = 1

print('\nVoxels after backprojection:\n')
bp = FanBackprojection()
volume = bp(volume, sino, angles)
print(volume.data)
plt.figure()
plt.imshow(cp.asnumpy(volume.data))
plt.colorbar()
plt.show()
