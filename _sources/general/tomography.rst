.. _principles:

======================
Tomographic projectors
======================

The discretized reconstruction problem of CT can be stated as retrieving the 3D object :math:`\mathbf x \in \mathbb{R}^{N_x N_y N_z}` discretized on a voxel grid by solving the linear inverse problem

.. math::

  \mathbf y = \mathbf A \mathbf x

where the forward projection :math:`\mathbf A \colon \mathbf x \mapsto \mathbf y` represents the X-ray transform, and :math:`\mathbf y \in \mathbb{R}^{N_\theta N_u N_v}` denotes the measured projections at :math:`N_\theta` angles of an :math:`(N_u, N_v)`-sized detector. The adjoint, :math:`\mathbf A^T \colon \mathbf y \mapsto \mathbf x`, is the backprojection. In practice, :math:`\mathbf y` is often not the direct quantity measured by a detector in an X-ray setup, but is instead obtained after several preprocessing steps of the raw measurement data (e.g., using a :math:`\log`-transform to invert Beer-Lambert's law\ :footcite:p:`kak_slaney_2001`).

Several discretization strategies can be considered to construct :math:`\mathbf A` as well as its transpose :math:`\mathbf A^T`:footcite:p:`hansen_2021`\ :footcite:p:`xu_2006`. Each of the :math:`N_u N_vN_\theta` rows of :math:`\mathbf A`, c.q.\ columns of :math:`\mathbf A^T`, discretizes a single line integral associated with the X-ray transform. That is, it chooses interpolation weights to approximate the integral

.. math::

   [\mathcal Ax]_{u, v, \theta} = \int_{-\infty}^\infty x\left(s_\theta + (d_{\theta, u, v} - s_\theta)t\right)\,\text{d}t,

which describes the straight line from a point source :math:`s_\theta` to a detector pixel midpoint :math:`d_{\theta, u, v}` through the volume :math:`x`. For 3D projectors, ASTRA estimates the line integral above with the Joseph kernel\ :footcite:p:`joseph1982`. In this *ray-driven* approach, each integration point takes a trilinearly interpolated value from neighboring points on the voxel grid. Importantly, the line is modeled such that it precisely arrives at a detector pixel's midpoint, and, hence, no re-interpolation is needed in the sinogram. Conversely, during a *voxel-driven* backprojection, a bilinear interpolation at each angle of the sinogram sums up to the voxel's value\ :footcite:p:`papenhausen_2011`. In this case, all lines of backprojection go precisely through the voxel's center, and now re-interpolation is avoided in the volume. The reader is referred to :footcite:t:`hansen_2021` (Chapter 9) for interpolation formulae.

Due to the difference in forward and backward lines, the 3D conebeam FP and BP projectors in ASTRA are *unmatched*, i.e., the backprojector is not the exact transpose of the forward projector. The approach, however, is advantageous for an implementation on GPUs. All parallel threads (discussed in Section \ref{sec:kernel}) are independent of each other, which avoids potential race conditions, i.e., when two threads would write to the same memory simultaneously\ :footcite:p:`despres_2017`. On the other hand, unmatched projectors lead to nonconvergence in iterative algorithms due to a nonsymmetry of the iteration matrix\ :footcite:p:`dong_2019`. In the presence of noise, this does not always pose a problem\ :footcite:p:`zeng_2000`.

.. footbibliography::
