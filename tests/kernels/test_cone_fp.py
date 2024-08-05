import itertools

import cupy as cp
import kernelkit as kk
import numpy as np
import pytest

from kernelkit import (ProjectionGeometry, ProjectionGeometrySequence, Beam,
    rotate, Detector, resolve_volume_geometry)
from kernelkit.kernels import RayDrivenConeFP
from kernelkit.kernels.reference_cone_fp import ReferenceConeFP
from kernelkit.kernel import copy_to_texture, KernelNotCompiledException, \
    KernelMemoryUnpreparedException


@pytest.fixture
def volume():
    return cp.ones((2, 2, 2), cp.float32)


@pytest.fixture
def volume_geometry(volume):
    return kk.resolve_volume_geometry(
        shape=volume.shape,
        extent_min=[-.5, None, None],
        extent_max=[.5, None, None])


@pytest.fixture
def geom_t0(
        rows=9,
        cols=9,
        src_dist=5.,
        det_dist=5.,
        px_h=1.,
        px_w=1.
):
    return kk.ProjectionGeometry(
        source_position=[-src_dist, 0, 0],
        detector_position=[det_dist, 0, 0],
        beam=Beam.CONE,
        detector=kk.Detector(rows, cols, px_h, px_w))


def test_call_raises_not_compiled_exception(geom_t0, volume, volume_geometry):
    K = RayDrivenConeFP()
    assert not K.is_compiled

    projs = cp.zeros((1, geom_t0.detector.rows, geom_t0.detector.cols),
                     dtype=cp.float32)
    ptrs = cp.asarray([p.data.ptr for p in projs])
    pg = ProjectionGeometrySequence.fromList([geom_t0])
    vol_txt = copy_to_texture(volume)
    with pytest.raises(KernelNotCompiledException):
        K.__call__(vol_txt, pg, volume_geometry, ptrs)
    assert not K.is_compiled
    K.compile()
    assert K.is_compiled


def test_call_raises_exception_when_parameters_not_set(geom_t0, volume,
                                                       volume_geometry):
    K = RayDrivenConeFP()
    K.compile()
    vol_txt = copy_to_texture(volume)
    projs = cp.zeros((1, geom_t0.detector.rows, geom_t0.detector.cols),
                     dtype=cp.float32)
    ptrs = cp.asarray([p.data.ptr for p in projs])
    pg = ProjectionGeometrySequence.fromList([geom_t0])
    with pytest.raises(KernelMemoryUnpreparedException):
        K.__call__(vol_txt, pg, volume_geometry, ptrs)


@pytest.mark.parametrize('axis', (0, 1, 2,))
def test_projection_axes(axis):
    """Tests that the kernel projects in the right direction, and that the
    projected square occurs on the right spot on the detector.

    Parameters
    ----------
    axis : int
        The axis to test.
    """
    # set up a 2x2cm detector
    det = Detector(
        rows=128,
        cols=128,
        pixel_height=8. / 128,
        pixel_width=8. / 128)

    # magnification factor 1.0
    SOD = 1e6
    SDD = 1.0 * SOD

    if axis == 0:
        gt = cp.zeros((1, 100, 100), dtype=cp.float32)
        gt[..., 80:100, 90:100] = 1.0
        geom_t0 = ProjectionGeometry(
            source_position=[-SOD, 0.0, 0.0],
            detector_position=[SDD - SOD, 0.0, 0.0],
            u=[0, 1, 0],
            v=[0, 0, 1],
            detector=det,
            beam=Beam.CONE)
        scan = [geom_t0]
    elif axis == 1:
        gt = cp.zeros((100, 1, 100), dtype=cp.float32)
        gt[80:100, ..., 90:100] = 1.0
        geom_t0 = ProjectionGeometry(
            source_position=[0.0, -SOD, 0.0],
            detector_position=[0.0, SDD - SOD, 0.0],
            u=[1, 0, 0],
            v=[0, 0, 1],
            detector=det,
            beam=Beam.CONE)
        scan = [geom_t0]
    elif axis == 2:
        gt = cp.zeros((100, 100, 1), dtype=cp.float32)
        gt[80:100, 90:100] = 1.0  # stretched in the x-dir
        # 1. source-det are aligned on x-axis
        # 2. rotation around the y axis ('pitch') with np.pi, so source moves
        # on top, det to bottom
        # 4. looking from the bottom, the +v=-z axis points to -x
        # (and +u=-y points still to -y).
        # 5. hence, (100, 10, 1) (a volume stretched in x) appears as
        # a vertical narrow volume in v. (vertical on the detector)
        geom_t0 = ProjectionGeometry(
            source_position=[0.0, 0.0, -SOD],
            detector_position=[0.0, 0.0, SDD - SOD],
            u=[1, 0, 0],  # in the x-direction
            v=[0, 1, 0],  # in the y-direction
            detector=det,
            beam=Beam.CONE)
        scan = [geom_t0]
    else:
        raise ValueError

    pg = ProjectionGeometrySequence.fromList(scan)
    # Check that the kernel is projecting in the expected direction
    np.testing.assert_equal(RayDrivenConeFP._geom_axes(pg)[0], axis)

    # an isotropic volume
    vg = resolve_volume_geometry(
        shape=gt.shape,
        voxel_size=[0.05] * 3
    )

    K = RayDrivenConeFP()
    K.compile()
    K.set_params(K.geoms2params(pg, vg))
    projs = cp.zeros((len(pg), pg[0].detector.rows, pg[0].detector.cols),
                     dtype=cp.float32)
    ptrs = cp.asarray([
        p.data.ptr for p in projs
    ])
    gt_txt = copy_to_texture(gt)
    K.__call__(gt_txt, pg, vg, ptrs, projs.shape)

    # # to test if pixels are appearing in the right quadrant of the detector,
    # # we just remove the irrelevant detector part
    out = np.asarray(projs[0].get() > 0.00000001, dtype=np.int64)
    out[:64, :] = 0.0

    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.imshow(projs[0].get())
    # plt.show()

    np.testing.assert_equal(np.sum(out), 128)


@pytest.mark.parametrize('nr_voxels', (1, 7, 100))
def test_invariant_under_voxel_size(nr_voxels):
    """Tests that the kernel produces the same integral value of
    a 1x1x1 volume, regardless of discretization.

    Parameters
    ----------
    nr_voxels : int
        The number of nr_voxels to discretize the volume in one dimension.
    """
    # set up a super small detector
    det = Detector(
        rows=1,
        cols=1,
        pixel_height=1. / 1024,
        pixel_width=1. / 1024)

    # magnification factor 1.1
    SOD = 4.
    SDD = 1.1 * SOD
    geom_t0 = ProjectionGeometry(
        source_position=[-SOD, 0.0, 0.0],
        detector_position=[SDD - SOD, 0.0, 0.0],
        detector=det,
        beam=Beam.CONE
    )
    scan = [geom_t0]
    gt = cp.ones([nr_voxels] * 3, dtype=cp.float32)
    pg = ProjectionGeometrySequence.fromList(scan)

    # an isotropic volume
    vg = resolve_volume_geometry(
        shape=gt.shape,
        extent_min=[-.5] * 3,
        extent_max=[.5] * 3
    )

    K = RayDrivenConeFP()
    K.compile()
    K.set_params(K.geoms2params(pg, vg))
    projs = cp.zeros((len(pg), pg[0].detector.rows, pg[0].detector.cols),
                     dtype=cp.float32)
    ptrs = cp.asarray([p.data.ptr for p in projs])
    gt_txt = copy_to_texture(gt)
    K.__call__(gt_txt, pg, vg, ptrs, projs.shape)

    volume_integral_through_center = 1.0
    np.testing.assert_almost_equal(
        projs[0, 0, 0].get(),
        volume_integral_through_center)


@pytest.mark.parametrize('axes, direction',
                         itertools.product(
                             itertools.permutations((0, 1, 2)),
                             (0, 1, 2)))
def test_volume_axes_choices(axes: tuple, direction: int):
    """Tests that the kernel interpolates correctly from any axis ordering
    of the volume. To do so, we create a volume with a 1.0-valued line
    through where the ray is going and random values everywhere else.

    The kernel should take the sum of the 1.0 values and not see any of the
    random other values.

    Note to get a precise value we need odd discretized detector and volume.

    Parameters
    ----------
    axes : tuple
        The axes of the volume to test. (0, 1, 2) corresponds to x, y, z in
        world coordinates in a right-handed coordinate system.
    """
    # set up a super small detector
    det = Detector(
        rows=9,
        cols=9,
        pixel_height=1. / 128,
        pixel_width=1. / 128)

    # magnification factor 1.0
    SOD = 4.
    SDD = 1.1 * SOD
    geom_t0 = ProjectionGeometry(
        source_position=[-SOD, 0.0, 0.0],
        detector_position=[SDD - SOD, 0.0, 0.0],
        detector=det,
        beam=Beam.CONE
    )

    # gt = np.random.random(size=[9] * 3) * 1000
    gt = np.zeros([9] * 3, dtype=cp.float32)

    if direction == 0:
        gt[:, gt.shape[1] // 2, gt.shape[2] // 2] = 1.
    elif direction == 1:
        geom_t0 = kk.rotate(geom_t0, yaw=np.pi / 2)
        gt[gt.shape[1] // 2, :, gt.shape[2] // 2] = 1.
    elif direction == 2:
        geom_t0 = kk.rotate(geom_t0, pitch=np.pi / 2)
        gt[gt.shape[1] // 2, gt.shape[2] // 2] = 1.

    scan = [geom_t0]
    gt = cp.asarray(gt, dtype=cp.float32)
    gt_T = cp.ascontiguousarray(cp.transpose(gt, axes))

    pg = ProjectionGeometrySequence.fromList(scan)
    np.testing.assert_equal(RayDrivenConeFP._geom_axes(pg)[0], direction)
    vg = resolve_volume_geometry(
        shape=gt.shape,
        extent_min=[-.5] * 3,
        extent_max=[.5] * 3
    )

    K = RayDrivenConeFP(volume_axes=axes)
    K.compile()
    K.set_params(K.geoms2params(pg, vg))
    projs = cp.zeros((len(pg), pg[0].detector.rows, pg[0].detector.cols),
                     dtype=cp.float32)
    ptrs = cp.asarray([p.data.ptr for p in projs])
    gt_T_txt = copy_to_texture(gt_T)
    K.__call__(gt_T_txt, pg, vg, ptrs, projs.shape)

    # plt.figure()
    # plt.title(f"Vol axes:{axes}, Ray dir: {direction}")
    # plt.imshow(projs[0].get())
    # plt.show()

    volume_integral_through_center = 1.0
    np.testing.assert_almost_equal(
        projs[0, det.rows // 2, det.cols // 2].get(),
        volume_integral_through_center)


projection_axes_expected_outcome = None


@pytest.mark.parametrize('axes',
                         itertools.permutations((0, 1, 2)))
def test_projection_axes_choices(axes: tuple):
    """Tests that the kernel writes correctly from any axis ordering

    Parameters
    ----------
    axes : tuple
        The axes of the volume to test. (0, 1, 2) corresponds to the number
        of angles, rows, columns of the projection data set.
    """
    N_theta, N_v, N_u = 7, 11, 13
    det = Detector(
        rows=N_v,
        cols=N_u,
        pixel_height=4. / 121,
        pixel_width=4. / 121)

    # magnification factor 1.0
    SOD = 4.
    SDD = SOD
    geom_t0 = ProjectionGeometry(
        source_position=[-SOD, 0.0, 0.0],
        detector_position=[SDD - SOD, 0.0, 0.0],
        detector=det,
        beam=Beam.CONE
    )
    scan = [
        rotate(
            geom_t0,
            yaw=i * 2 * np.pi / N_theta
        ) for i in range(N_theta)
    ]

    gt = cp.zeros((10, 10, 10), dtype=cp.float32)
    gt[2:3] = 1.
    gt[:, 2:3] = 2.
    gt[:, :, 2:3] = 3.

    pg = ProjectionGeometrySequence.fromList(scan)
    vg = resolve_volume_geometry(
        shape=gt.shape,
        voxel_size=0.02
    )

    K = RayDrivenConeFP(projection_axes=axes)
    K.compile()
    K.set_params(K.geoms2params(pg, vg))

    projs = cp.zeros((len(pg), pg[0].detector.rows, pg[0].detector.cols),
                     dtype=cp.float32)
    projs_transposed = cp.ascontiguousarray(cp.transpose(projs, axes))
    assert projs_transposed.flags.c_contiguous
    ptrs = cp.asarray([p.data.ptr for p in projs_transposed])
    gt_txt = copy_to_texture(gt)
    K.__call__(gt_txt, pg, vg, ptrs)

    # reverse the transposition
    reverse_transpose_axes = tuple(np.argsort(axes))
    projs[...] = cp.transpose(projs_transposed, reverse_transpose_axes)

    global projection_axes_expected_outcome
    if projection_axes_expected_outcome is None:
        projection_axes_expected_outcome = np.sum(projs[0].get())
    else:
        np.testing.assert_almost_equal(
            projection_axes_expected_outcome, np.sum(projs[0].get()))


def test_reference_cone_fp():
    raise NotImplementedError()
    vol = cp.ones((4, 4, 4), cp.float32)
    proj = cp.zeros((9, 9), dtype=cp.float32)

    SOD = 50.0
    SDD = 100.0

    det = kk.Detector(9, 9, 1., 1.)
    pg = kernelkit.CircularGeometry(
        angles=np.linspace(0, 2 * np.pi, 1, endpoint=False),
        detector=det,
        source_object_distance=SOD,
        source_detector_distance=SDD,
        beam=kernelkit.Beam.CONE
    )

    vg = kk.resolve_volume_geometry(
        shape=vol.shape,
        extent_min=np.array((-.5, -.5, -.5)),
        extent_max=np.array((.5, .5, .5))
    )
    vg.check_consistency()

    fpkern = ReferenceConeFP()
    fpkern.compile()
    fpkern.__call__(vol, proj, pg, vg)
    proj_cpu = proj.get()
    assert proj_cpu.shape == proj.shape
    assert proj_cpu[4, 4] == pytest.approx(1., 1e-6)

    proj.fill(0.)
    vol_txt = copy_to_texture(vol)
    proj_ptrs = cp.asarray(
        [p.data.ptr for p in [proj, ]]
    )
    pg = [kk.ProjectionGeometry(
        [-SOD, 0.0, 0.0],
        [SDD - SOD, 0.0, 0.0],
        det),
    ]
    spg = ProjectionGeometrySequence.fromList(pg)

    fpkern = RayDrivenConeFP()
    fpkern.compile()
    params = fpkern.geoms2params(spg, vg)
    fpkern.set_params(params)
    fpkern.__call__(
        vol_txt,
        spg,
        vg,
        proj_ptrs,
    )
    # volume_geometry=vol_geom,
    # projection_geometry=kk.ProjectionGeometrySequence.fromList([geom()]),
    # projections=)
    proj_cpu = proj.get()
    assert proj_cpu.shape == proj.shape
    assert proj_cpu[4, 4] == pytest.approx(1., 1e-6)


def test_pitched_projections():
    raise NotImplementedError()
    volume = cp.ones((100, 100, 1), dtype=cp.float32)
    volume_geometry = kk.VolumeGeometry(
        shape=volume.shape,
        voxel_size=(1., 1., 1.),
        extent_min=[-0.5, -0.5, -0.5],
        extent_max=[0.5, 0.5, 0.5])

    proj = cp.ones((100, 100), dtype=cp.float32)
    assert not kk.ispitched(proj)
    proj = kk.aspitched(proj)

    # let's screw up the padded part of the pitched projection
    proj.base.fill(1e6)
    proj.fill(0.)

    kern = kk.kernels.RayDrivenConeFP()
    kern.compile()
    params = kern.geoms2params([kk.ProjectionGeometry(
        source_position=[-100000000., 0., 0.],  # parallel beam
        detector_position=[0., 0., 0.],
        detector=kk.Detector(rows=proj.shape[0],
                             cols=proj.shape[1],
                             pixel_width=1. / proj.shape[1],
                             pixel_height=1. / proj.shape[1]))],
        volume_geometry)

    kern.set_params(params)
    texture = kk.kernel.copy_to_texture(proj, texture_type="pitch2d")
    pointer_array = cp.asarray([texture.ptr])
    kern(pointer_array, volume, volume_geometry)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(volume[..., 0].get())
    plt.show()