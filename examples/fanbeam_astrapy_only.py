"""Example using the ray transform with fan beam geometry."""

import matplotlib.pyplot as plt
import numpy as np
import odl

import astrapy

# Reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20],
                               shape=[100, 100], dtype='float32')

# Make a fan beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
# Detector: uniformly sampled, n = 512, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 512)
geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition,
                                    src_radius=1000, det_radius=100)

# Ray transform (= forward projection).
# ray_trafo_astrapy = odlmod.tomo.RayTransform(reco_space, geometry, impl='astrapy')
ray_trafo_astrapy = odl.tomo.RayTransform(reco_space, geometry,
                                          impl=astrapy.RayTrafoImpl)

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)
phantom.show()

# Create projection data by calling the ray transform on the phantom
proj_data_astrapy = ray_trafo_astrapy(phantom)
proj_data_astrapy.show(title='Projection Data (astrapy)')

# projection data (or any element in the projection space).
backproj_astrapy = ray_trafo_astrapy.adjoint(proj_data_astrapy)

# Shows a slice of the phantom, projections, and reconstruction
backproj_astrapy.show(title='Back-projection (astrapy)', vmin=0., vmax=100)
plt.show()
