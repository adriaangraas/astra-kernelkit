import itertools

import cupy as cp
import numpy as np
import pytest

import kernelkit as kk
from kernelkit.kernel import KernelMemoryUnpreparedException, \
    KernelNotCompiledException, copy_to_texture
from kernelkit.kernels import VoxelDrivenConeBP


@pytest.fixture
def projs():
    angles, rows, cols = 3, 9, 9
    return cp.zeros((angles, rows, cols), dtype=cp.float32)


@pytest.fixture
def volume():
    return cp.ones((2, 2, 2), cp.float32)


@pytest.fixture
def volume_geometry(volume):
    return kk.resolve_volume_geometry(volume.shape,
                                      [-.2, None, None], [.2, None, None])


@pytest.fixture
def geom(rows=9,
         cols=9,
         src_dist=5.,
         det_dist=5.,
         px_h=1.,
         px_w=1.):
    return kk.ProjectionGeometry(
        [-src_dist, 0, 0],
        [det_dist, 0, 0],
        kk.Detector(rows, cols, px_h, px_w))


def test_call_raises_not_compiled_exception(projs, volume, volume_geometry):
    kern = VoxelDrivenConeBP()
    assert not kern.is_compiled
    textures = kk.kernel.copy_to_texture(projs)
    with pytest.raises(KernelNotCompiledException):
        kern(textures, volume, volume_geometry)
    assert not kern.is_compiled


def test_call_raises_exception_when_parameters_not_set(projs, volume,
                                                       volume_geometry):
    kern = VoxelDrivenConeBP()
    kern.compile()
    textures = kk.kernel.copy_to_texture(projs)
    with pytest.raises(KernelMemoryUnpreparedException):
        kern(textures, volume, volume_geometry)


def test_single_pixel_detector_and_volume(projs, volume, volume_geometry):
    volume = cp.zeros((1, 1, 1), dtype=cp.float32)
    projs = cp.ones((1, 2, 2), dtype=cp.float32)
    proj_geom = kk.ProjectionGeometry(
        source_position=[-10000000000., 0., 0.],  # parallel beam
        detector_position=[0., 0., 0.],
        detector=kk.Detector(rows=projs.shape[1],
                             cols=projs.shape[2],
                             pixel_width=1./projs.shape[2],
                             pixel_height=1./projs.shape[1]))
    vol_geom = kk.VolumeGeometry(
        shape=volume.shape,
        voxel_size=(1., 1., 1.),
        extent_min=[-0.5, -0.5, -0.5],
        extent_max=[0.5, 0.5, 0.5])
    kern = VoxelDrivenConeBP()
    kern.compile()
    params = kern.geoms2params([proj_geom], vol_geom, with_adjoint_scaling=True)
    kern.set_params(params)
    textures = kk.kernel.copy_to_texture(projs)
    kern(textures, volume, volume_geometry)


@pytest.mark.parametrize('axis', (0, 1, 2, ))
def test_backprojection_axes(axis):
    """Tests that the kernel backprojects in the right direction, and that the
    backprojected line occurs in the right place in the volume.

    Parameters
    ----------
    axis : int
        The axis to test.
    """
    # set up a 1x1cm detector
    det = kk.Detector(
        rows=1,
        cols=1,
        pixel_height=1.,
        pixel_width=1.)

    # magnification factor 1.0
    SOD = 100000.  # close to parallel beam
    SDD = 1.0 * SOD
    geom_t0 = kk.ProjectionGeometry(
        source_position=[-SOD, 0.0, 0.0],
        detector_position=[SDD - SOD, 0.0, 0.0],
        detector=det,
        beam=kk.Beam.CONE
    )

    vol = cp.zeros((3, 3, 3), dtype=cp.float32)  # x, y, z
    expected = np.zeros((3, 3, 3), dtype=np.float32)

    if axis == 0:
        scan = [kk.rotate(geom_t0, yaw=0.0)]
        expected[:, 1, 1] = 1.  # a ray through the empty volume at axis 0
    elif axis == 1:
        scan = [kk.rotate(geom_t0, yaw=np.pi / 2)]
        expected[1, :, 1] = 1.  # a ray through the empty volume at axis 1
    elif axis == 2:
        scan = [kk.rotate(geom_t0, pitch=np.pi / 2)]
        expected[1, 1, :] = 1.  # a ray through the empty volume at axis 2
    else:
        raise ValueError

    pg = kk.ProjectionGeometrySequence.fromList(scan)

    # an isotropic volume with voxel sizes of 1.
    vg = kk.resolve_volume_geometry(
        shape=vol.shape,
        voxel_size=(1., 1., 1.),
    )
    vg.check_consistency()

    K = VoxelDrivenConeBP()
    K.compile(texture=K.InterpolationMethod.Tex2DLayered)
    K.set_params(K.geoms2params(pg, vg))
    projs = cp.ones((len(pg), det.rows, det.cols), dtype=cp.float32)
    projs_txt = copy_to_texture(projs, layered=True)
    K.__call__(projs_txt, vol, vg)

    np.testing.assert_array_almost_equal(vol.get(), expected, decimal=4)

volume_axes_expected_outcome = None


@pytest.mark.parametrize('axes',
                         itertools.permutations((0, 1, 2)))
def test_volume_axes_choices(axes: tuple):
    """Tests that the kernel writes correctly from any axis ordering

    Parameters
    ----------
    axes : tuple
        The axes of the volume to test. (0, 1, 2) corresponds to the number
        of angles, rows, columns of the projection data set.
    """
    N_theta, N_v, N_u = 1, 20, 20
    det = kk.Detector(
        rows=N_v,
        cols=N_u,
        pixel_height=1.,
        pixel_width=1.)

    # magnification factor 1.0
    SOD = 1000.
    SDD = SOD
    geom_t0 = kk.ProjectionGeometry(
        source_position=[-SOD, 0.0, 0.0],
        detector_position=[SDD - SOD, 0.0, 0.0],
        detector=det,
        beam=kk.Beam.CONE
    )
    scan = [
        kk.rotate(
            geom_t0,
            yaw=i * 2 * np.pi / N_theta
        ) for i in range(N_theta)
    ]

    vol = cp.zeros((60, 60, 60), dtype=cp.float32)

    vol.flags.writeable = False
    pg = kk.ProjectionGeometrySequence.fromList(scan)
    vg = kk.resolve_volume_geometry(
        shape=vol.shape,
        voxel_size=0.5,
    )

    K = VoxelDrivenConeBP(volume_axes=axes)
    K.compile(texture=K.InterpolationMethod.Tex2DLayered)
    K.set_params(K.geoms2params(pg, vg))
    projs = cp.ones((len(pg), pg[0].detector.rows, pg[0].detector.cols),
                     dtype=cp.float32)
    projs_txt = copy_to_texture(projs, layered=True)

    vol_transposed = cp.ascontiguousarray(cp.transpose(vol, axes))
    K.__call__(projs_txt, vol_transposed, vg)

    # reverse the transposition
    reverse_transpose_axes = tuple(np.argsort(axes))
    vol[...] = cp.transpose(vol_transposed, reverse_transpose_axes)
    assert vol.shape == vg.shape

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 3)
    # axs[0].imshow(vol[vol.shape[0] // 2].get())
    # axs[1].imshow(vol[:, vol.shape[1] // 2].get())
    # axs[2].imshow(vol[..., vol.shape[2] // 2].get())
    # plt.show()

    outcome = vol[..., vol.shape[2] // 2].get()  # central slice
    global volume_axes_expected_outcome
    if volume_axes_expected_outcome is None:
        volume_axes_expected_outcome = outcome
    else:
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(vol[..., vol.shape[2] // 2].get() - volume_axes_expected_outcome)
        # plt.show()
        np.testing.assert_almost_equal(
            volume_axes_expected_outcome,
            outcome,
            decimal=3)



def test_pitched_projections():
    volume = cp.zeros((100, 100, 1), dtype=cp.float32)
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
    proj.fill(1.)


    kern = VoxelDrivenConeBP()
    kern.compile(texture=kern.InterpolationMethod.Tex2D)
    params = kern.geoms2params([kk.ProjectionGeometry(
        source_position=[-100000000., 0., 0.],  # parallel beam
        detector_position=[0., 0., 0.],
        detector=kk.Detector(rows=proj.shape[0],
                             cols=proj.shape[1],
                             pixel_width=1./proj.shape[1],
                             pixel_height=1./proj.shape[1]))],
        volume_geometry)

    kern.set_params(params)
    texture = kk.kernel.copy_to_texture(proj, texture_type="pitch2d")
    pointer_array = cp.asarray([texture.ptr])
    kern(pointer_array, volume, volume_geometry)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(volume[..., 0].get())
    plt.show()


