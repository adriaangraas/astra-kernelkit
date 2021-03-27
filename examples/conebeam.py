import os

import matplotlib.pyplot as plt
import numpy as np
import odl
import os

from odl.tomo import fbp_filter_op, fbp_op

from astrapy.odlsupport import RayTrafoImpl

np.set_printoptions(suppress=True)
"""
    Set `NO_ASTRAPY` to run with classic ODL + ASTRA.
    Keep unset to run with astrapy + CuPy.
"""
try:
    no_astrapy = int(os.environ.get('NO_ASTRAPY')) == 1
except:
    no_astrapy = False

if not no_astrapy:
    import astrapy


# Reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
# reco_space = odlsupport.uniform_discr(min_pt=[-10, -20, -30], max_pt=[10, 20, 30],
#                                shape=[10,20,30], dtype='float32')
y = 200
# reco_space = odlsupport.uniform_discr(min_pt=[-3*y, -2*y, -y], max_pt=[3*y, 2*y, y],
#                                shape=[3*y, 2*y, y], dtype='float32')
x = 80
reco_space = odl.uniform_discr(min_pt=[-x//2, -x, -x*2], max_pt=[x//2, x, x*2],
                               shape=[y//2, y, y*2], dtype='float32')
# reco_space = odl.uniform_discr(min_pt=[-2*x, -x, -x], max_pt=[2*x, x, x],
#                                shape=[2*y, y, y], dtype='float32')
# reco_space = odl.uniform_discr(min_pt=[-x, -x, -x], max_pt=[x, x, x],
#                                shape=[y, y, y], dtype='float32')

nr_angles = 5000
angle_partition = odl.uniform_partition(0, 2 * np.pi, nr_angles)
# Detector uniformly sampled, n = 512, min = -30, max = 30
q = 123.45
r = 128*2
detector_partition = odl.uniform_partition(min_pt=[-q, -2*q],
                                           max_pt=[q, 2*q],
                                           shape=[r, 2*r]) # u, v

geometry = odl.tomo.ConeBeamGeometry(angle_partition, detector_partition,
                                     src_radius=150, det_radius=50)

# Ray transform (= forward projection).
if no_astrapy:
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
else:
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry,
                                      impl=RayTrafoImpl)

# Create a discrete Shepp-Logan phantom (modified /1version)
np.random.seed(0)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)
rand = np.random.random_sample(phantom.data.shape)
# phantom.data[:] *= rand
# phantom.data[:] = 1.

# phantom.show()
# plt.figure()
# plt.imshow(phantom.data[..., 25])
# plt.show()

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)
# proj_data_astrapy.show(title='Projection Data (astrapy)')
# plt.show()
d = proj_data.data
# proj_data_astrapy.data[:] = 100.
# d = np.transpose(d, [0,2,1])
# d = np.ascontiguousarray(d)
# print(d.shape)

plt.figure()
plt.imshow(d[0])
if d.shape[0] > 1:
    plt.figure()
    plt.imshow(d[1])
if d.shape[0] > 2:
    plt.figure()
    plt.imshow(d[2])
plt.show()

# op2 = fbp_op(ray_trafo)
# yes = op2(proj_data)
# plt.figure()
# plt.imshow(yes[:, y//2, :])
# plt.show()


# # projection data (or any element in the projection space).
backproj_astrapy = ray_trafo.adjoint(proj_data)
# # backproj_astrapy.show()
v = backproj_astrapy.data
# print(v.shape)
# # v = np.transpose(v, [2,1,0])
plt.figure()
# # plt.imshow(np.concatenate(v).reshape(50*5, 50*10))
# # plt.imshow(np.vstack([v[..., i] for i in range(v.shape[2])]))
plt.imshow(v[:, y//2, :])
# # plt.figure()
# # plt.imshow(v[:, :, z//2])
# # plt.figure()
# # plt.imshow(v[:, :, z//2+2])
plt.show()
#
# # Shows a slice of the phantom, projections, and reconstruction
# # backproj_astrapy.show(title='Back-projection (astrapy)')
# # plt.show()
