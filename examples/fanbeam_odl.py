"""Example using the ray transform with fan beam geometry."""

import numpy as np
import odl
import astrapy.odlmod as odlmod

# Reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='float32')

# Make a fan beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
angle_partition = odl.uniform_partition(0, 2 * np.pi, 360)
# Detector: uniformly sampled, n = 512, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 512)
geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition,
                                    src_radius=1000, det_radius=100)

# Ray transform (= forward projection).
ray_trafo_astrapy = odlmod.tomo.RayTransform(reco_space, geometry, impl='astrapy')
ray_trafo_astra_cuda = odl.tomo.RayTransform(reco_space, geometry, impl='astra_cuda')

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create projection data by calling the ray transform on the phantom
proj_data_astrapy = ray_trafo_astrapy(phantom)
proj_data_astra_cuda = ray_trafo_astra_cuda(phantom)

vmin = min(np.min(proj_data_astrapy), np.min(proj_data_astra_cuda))
vmax = max(np.max(proj_data_astrapy), np.max(proj_data_astra_cuda))
proj_data_astrapy.show(title='Projection Data (astrapy)', vmin=vmin, vmax=vmax)
proj_data_astra_cuda.show(title='Projection Data (astra_cuda)', vmin=vmin, vmax=vmax, force_show=True)
# projection data (or any element in the projection space).
backproj_astrapy = ray_trafo_astrapy.adjoint(proj_data_astrapy)
backproj_astra_cuda = ray_trafo_astra_cuda.adjoint(proj_data_astra_cuda)

# Shows a slice of the phantom, projections, and reconstruction
# phantom.show(title='Phantom')
backproj_astrapy.show(title='Back-projection (astrapy)', vmin=0., vmax=100)
backproj_astra_cuda.show(title='Back-projection (astra_cuda)', force_show=True, vmin=0., vmax=100)
