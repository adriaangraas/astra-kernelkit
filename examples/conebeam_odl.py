import os

import numpy as np
import odl

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
    pass

sz = 80
vxls = 200
reco_space = odl.uniform_discr(min_pt=[-sz // 2, -sz, -sz * 2],
                               max_pt=[sz // 2, sz, sz * 2],
                               shape=[vxls // 2, vxls, vxls * 2],
                               dtype='float32')
nr_angles = 500
angle_partition = odl.uniform_partition(0, 2 * np.pi, nr_angles)
q = 123.45
cols, rows = 256, 128
detector_partition = odl.uniform_partition(min_pt=[-q, -2 * q],
                                           max_pt=[q, 2 * q],
                                           shape=[cols, rows])  # u, v
geometry = odl.tomo.ConeBeamGeometry(angle_partition, detector_partition,
                                     src_radius=150, det_radius=50)

# Ray transform (= forward)
if no_astrapy:
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
else:
    ray_trafo = odl.tomo.RayTransform(reco_space, geometry,
                                      impl=RayTrafoImpl)
    ray_trafo_filter = odl.tomo.RayTransform(reco_space, geometry,
                                      impl=RayTrafoImpl)
    ray_trafo_filter.get_impl().filter = True

# Create a discrete Shepp-Logan phantom (modified /1version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)
d = proj_data.data
proj_data[0].show()

# CuPy Raytrafo + ODL filter doesn't go well a.t.m., not sure why
from odl.tomo import fbp_op
backproj_astrapy = fbp_op(ray_trafo)(proj_data)
backproj_astrapy.show(title='Back-projection with ODL filter op')

if not no_astrapy:
    backproj_astrapy = ray_trafo_filter.adjoint(proj_data)
    backproj_astrapy.show(title='Back-projection with CuPy filter')

backproj_astrapy = ray_trafo.adjoint(proj_data)
backproj_astrapy.show(title='Back-projection pure')

import matplotlib.pyplot as plt
plt.show()
