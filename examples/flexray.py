from typing import Any

import cupy as cp
import numpy as np
import reflex
from reflex import centralize

from astrapy import bp, suggest_volume_extent
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


def reconstruct(
    settings_path,
    projs_path,
    darks_path,
    flats_path,
    voxels_x,
    proj_ids,
    nr_angles_per_rot=None,
    rot=2 * np.pi,
    reco_interval=None,
    plot_pyqtgraph=False,
    corrections: Any = True,
    mode_slice: Any = False,
    vol_scaling_factor: float = 1.,
):
    settings = reflex.Settings.from_path(settings_path)
    if nr_angles_per_rot is None:
        nr_angles_per_rot = len(proj_ids)
    angles = proj_ids * rot / nr_angles_per_rot
    if reco_interval is None:
        reco_interval = int(np.ceil(2 * np.pi * nr_angles_per_rot / rot))

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

    vol_min, vol_max = suggest_volume_extent(geometry[0])
    if mode_slice is False:
        projections = reflex.projs(projs_path, proj_ids)
        vol = bp(projections,
                 geometry,
                 (voxels_x, None, None),
                 np.array(vol_min) * vol_scaling_factor,
                 np.array(vol_max) * vol_scaling_factor,
                 chunk_size=50,
                 fpreproc=_preproc_projs)

        if plot_pyqtgraph:
            import pyqtgraph as pq
            pq.image(vol)
            import matplotlib.pyplot as plt
            plt.figure()
            plt.imshow(vol[..., vol.shape[2] // 2])
            plt.show()
    else:
        def _fload(ids):
            return reflex.projs(projs_path, ids, verbose=False)

        from astrapy.kernels import ConeBackprojection
        bufferbp = ProjectionBufferBp(
            ConeBackprojection(),
            proj_ids,
            geometry,
            _fload,
            # filter=None,
            fpreproc=_preproc_projs)

        import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.gca()
        # axim = ax.imshow(np.ones((vxls_x, vxls_x)), vmin=-1e4, vmax=1e4)
        # axim = ax.imshow(np.ones((vxls_x, vxls_x)), vmin=-8., vmax=20.)
        # axim = ax.imshow(np.ones((vxls_x, vxls_x)), vmin=0, vmax=.005)
        # axim = ax.imshow(np.ones((vxls_x, vxls_x)), vmin=-100., vmax=2000.)

        state = iter(range(0, len(proj_ids) - reco_interval))
        w, h = vol_max[0], vol_max[2]
        def runf():
            n = next(state)
            if mode_slice == 'temporal':
                r = range(n, n + reco_interval)
            else:
                r = range(0, len(proj_ids))
            return bufferbp.slice(
                r,
                slice_shape=(voxels_x, voxels_x),
                slice_extent_min=np.array((-w, -w)) * 1.5,
                slice_extent_max=np.array((w, w)) * 1.5,
                slice_rotation=[n / 50 * np.pi] * 3,
                slice_position=np.array(
                    (np.sin(n / 50), np.cos(n / 50), np.sin(n / 50)))
                               * w / 2)

        # def runaf():
        #     return bufferbp.randslice(
        #         numbers_sampling_set=range(next(state) + reco_interval), #range(len(proj_ids)),
        #         numbers_sample_size=reco_interval,
        #         slice_shape=(voxels_x, voxels_x),
        #         slice_extent_min=(-w, -w),
        #         slice_extent_max=(w, w),
        #         slice_rotation=(2 * np.pi, 2 * np.pi, 2 * np.pi),
        #         slice_position_min=(-w / 4, -w / 4, -h / 4),
        #         slice_position_max=(w / 4, w / 4, h / 4))

        # axim.set_array(slice)
        # plt.pause(.001)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    prefix = "/home/adriaan/data"
    path = f"{prefix}/pink_plant"
    corrections = reflex.motor.Corrections()
    corrections.tra_det_correction = 24.
    corrections.tra_obj_correction = -.6
    reconstruct(
        settings_path=path,
        projs_path=path,
        darks_path=path,
        flats_path=path,
        voxels_x=400,
        # proj_range=range(0, 1201, 8),
        # angles=np.arange(0, 2 * np.pi, 2 * np.pi / 1201 * 8),
        proj_ids=np.arange(1201),
        plot_pyqtgraph=True,
        corrections=True,
        mode_slice=False,
        vol_scaling_factor=.7)
