"""Example using the ray transform with 2d parallel beam geometry."""

import numpy as np
import odlmod
from astrapy.odlmod.tomo.operators import modified_ray_trafo

# Reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odlmod.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='float32')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 180, min = 0, max = pi
angle_partition = odlmod.uniform_partition(0, np.pi, 180)

# Detector: uniformly sampled, n = 512, min = -30, max = 30
detector_partition = odlmod.uniform_partition(-30, 30, 512)

geometry = odlmod.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform (= forward projection).
ray_trafo = modified_ray_trafo.RayTransform(reco_space, geometry, impl='astrapy')

# Create a discrete Shepp-Logan phantom (modified version)
phantom = odlmod.phantom.shepp_logan(reco_space, modified=True)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)

# Back-projection can be done by simply calling the adjoint operator on the
# projection data (or any element in the projection space).
backproj = ray_trafo.adjoint(proj_data)

# Shows a slice of the phantom, projections, and reconstruction
phantom.show(title='Phantom')
proj_data.show(title='Projection Data (Sinogram)')
backproj.show(title='Back-projection', force_show=True)
