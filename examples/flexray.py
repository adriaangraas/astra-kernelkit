from typing import Any, Sequence

import cupy as cp
import numpy as np
import reflex
from reflex import centralize

from astrapy import bp
from astrapy.geom3d import AstraStatic3DGeometry, Flat2DDetector

def geom(settings, angles, corrections: Any = True, verbose=None):
    motor_geom = reflex.motor_geometry(
        settings,
        corrections,
        verbose=verbose)
    geom = reflex.circular_geometry(
        settings,
        initial_geom=motor_geom,
        angles=angles)

    geoms = []
    for i, (t, static_geom) in enumerate(geom.to_dict().items()):
        g = centralize(static_geom)
        hv = g.stage_rotation_matrix @ [0, 1, 0]
        vv = g.stage_rotation_matrix @ [0, 0, 1]
        g = reflex.centralize(static_geom)
        det = Flat2DDetector(
            rows=g.detector.rows,
            cols=g.detector.cols,
            pixel_width=g.detector.pixel_width,
            pixel_height=g.detector.pixel_height)
        geom = AstraStatic3DGeometry(
            tube_pos=g.tube_position,
            det_pos=g.detector_position,
            u_unit=hv,
            v_unit=vv,
            detector=det)
        geoms.append(geom)

    return geoms


def compute_volume_dimensions(settings, nr_voxels_x=None):
    det = reflex.SquarePixelDetector(
        settings.det_binned_rows(),
        settings.det_binned_cols(),
        settings.binned_pixel_size())

    # voxel width according to physical pixel size
    voxel_width, voxel_height = settings.suggested_voxel_size()
    # make voxel size larger if a small number of voxels is needed, and
    # vice versa
    if nr_voxels_x is None:
        nr_voxels_x = det.cols
        nr_voxels_z = det.rows
        scaling = 1
    else:
        # the amount of z-voxels should be proportional the detector dimensions
        estimated_nr_voxels_z = nr_voxels_x / det.cols * det.rows
        nr_voxels_z = int(np.ceil(estimated_nr_voxels_z))
        # since voxels_x is given, we need to scale the suggested voxel size
        scaling = nr_voxels_x / det.cols

    voxel_width /= scaling
    voxel_height /= scaling
    # compute total volume size based on the rounded voxel size
    width_from_center = voxel_width * nr_voxels_x / 2
    height_from_center = voxel_height * nr_voxels_z / 2
    return nr_voxels_x, nr_voxels_z, width_from_center, height_from_center


def reconstruct(
    settings_path,
    projs_path,
    darks_path,
    flats_path,
    voxels_x,
    proj_ids,
    nr_angles_per_rot=None,
    plot_pyqtgraph=False,
    corrections: Any = True,
    mode_slice: Any = False,
    vol_scaling_factor: float = 1.,
):
    settings = reflex.Settings.from_path(settings_path)
    if nr_angles_per_rot is None:
        nr_angles_per_rot = len(proj_ids)
    angles = proj_ids * (2 * np.pi) / nr_angles_per_rot

    darks = reflex.darks(darks_path)
    whites = reflex.flats(flats_path)

    if len(darks) != 0:
        if len(darks) > 1:
            darks = darks.mean(0)

        dark = cp.squeeze(cp.array(darks))

    if len(whites) != 0:
        if len(whites) > 1:
            whites = (whites - darks if len(darks) != 0 else whites).mean(0)

        white = cp.squeeze(cp.array(whites))

    def _preproc_projs(projs: cp.ndarray):
        # remove darkfield from projections
        cp.subtract(projs, dark, out=projs)
        cp.divide(projs, white, out=projs)  # flatfield the projections
        cp.log(projs, out=projs)  # take -log to linearize detector values
        return cp.multiply(projs, -1, out=projs)  # sunny side up

    geometry = geom(settings, angles, corrections)

    vxls_x, vxls_z, w, h = compute_volume_dimensions(settings, voxels_x)
    if mode_slice is False:
        projections = reflex.projs(projs_path, proj_ids)
        vol = bp(projections,
                 geometry,
                 (vxls_x, vxls_x, vxls_z),
                 np.array((-w, -w, -h)) * vol_scaling_factor,
                 np.array((w, w, h)) * vol_scaling_factor,
                 chunk_size=50,
                 fpreproc=_preproc_projs)

        if plot_pyqtgraph:
            import pyqtgraph as pq
            pq.image(vol)
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(vol[..., vxls_z // 2])
            plt.show()
    else:
        def _fload(ids):
            return reflex.projs(projs_path, ids)

        from astrapy.algo import ProjectionBufferBp
        from astrapy.kernels import ConeBackprojection
        bufferbp = ProjectionBufferBp(
            ConeBackprojection(),
            proj_ids,
            geometry,
            _fload,
            # filter=None,
            fpreproc=_preproc_projs)

        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.gca()
        # axim = ax.imshow(np.ones((vxls_x, vxls_x)), vmin=-1e4, vmax=1e4)
        # axim = ax.imshow(np.ones((vxls_x, vxls_x)), vmin=-8., vmax=20.)
        axim = ax.imshow(np.ones((vxls_x, vxls_x)), vmin=0, vmax=.5)
        # axim = ax.imshow(np.ones((vxls_x, vxls_x)), vmin=-100., vmax=2000.)

        for n in range(0, len(proj_ids) - 150):
            if mode_slice == 'temporal':
                r = range(n, n + 150)
            else:
                r = range(0, len(proj_ids))

            slice = bufferbp.slice(
                r,
                # slice_shape=(vxls_x, vxls_x),
                slice_shape=(vxls_x, vxls_x),
                slice_extent_min=(-w, -w),
                slice_extent_max=(w, w),
                slice_rotation=[n / 100 * np.pi] * 3,
                slice_position=np.array(
                    (np.sin(n / 100), np.cos(n / 100), np.sin(n / 100))) * w / 2
            )

            # slice = bufferbp.randslice(
            #     numbers_range=range(len(proj_range)),
            #     numbers_length=75,
            #     slice_shape=(vxls_x, vxls_x) ,
            #     slice_extent_min=(-w, -w),
            #     slice_extent_max=(w, w),
            #     slice_rotation=(2 * np.pi, 2 * np.pi, 2 * np.pi),
            #     slice_position_min=(-w/4, -w/4, -h/4),
            #     slice_position_max=(w/4, w/4, h/4),
            # )
            axim.set_array(slice)
            plt.pause(.0001)

        plt.show()
        #     plt.draw()
        #     plt.pause(0.01)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    prefix = "/home/adriaan/data"
    # # prefix = "/bigstore/felix/CT/TabletDissolve2/12ms4x4bin/handsoap_layeredSand"
    # path = f"{prefix}/handsoap_layeredSand/scan_4"
    # settings_path = f"{prefix}/handsoap_layeredSand/pre_scan"
    # reconstruct(
    #     projs_path=path,
    #     settings_path=settings_path,
    #     darks_path=settings_path,
    #     flats_path=settings_path,
    #     voxels_x=None,
    #     # proj_range=range(0, 1201),
    #     # angles=np.linspace(0, 2 * np.pi, 150, endpoint=False),
    #     proj_ids=np.arange(1000),
    #     nr_angles_per_rot=150,
    #     plot_pyqtgraph=True,
    #     corrections=True,
    #     mode_slice='temporal')

    path = f"{prefix}/pink_plant"
    corrections = reflex.motor.Corrections()
    corrections.tra_det_correction = 24.
    corrections.tra_obj_correction = -.6
    reconstruct(
        settings_path=path,
        projs_path=path,
        darks_path=path,
        flats_path=path,
        voxels_x=200,
        # proj_range=range(0, 1201, 8),
        # angles=np.arange(0, 2 * np.pi, 2 * np.pi / 1201 * 8),
        proj_ids=np.arange(1201),
        plot_pyqtgraph=True,
        corrections=True,
        mode_slice=False,
        vol_scaling_factor=.7)
