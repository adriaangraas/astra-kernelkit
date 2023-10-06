from typing import Any

import cupy as cp
import numpy as np
import reflex
from reflex import centralize

from astrapy import fdk, sirt_experimental, suggest_volume_extent
from astrapy.geom import Detector, ProjectionGeometry


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
        det = Detector(
            rows=g.detector.rows,
            cols=g.detector.cols,
            pixel_width=g.detector.pixel_width,
            pixel_height=g.detector.pixel_height)
        geom = ProjectionGeometry(
            source_position=g.source_position,
            detector_position=g.detector_position,
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
    plot_pyqtgraph=False,
    corrections: Any = True,
):
    settings = reflex.Settings.from_path(settings_path)
    if nr_angles_per_rot is None:
        nr_angles_per_rot = len(proj_ids)
    angles = proj_ids * rot / nr_angles_per_rot

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

    def _preproc(projs):
        xp = cp.get_array_module(projs[0])
        for p in projs:
            xp.subtract(p, dark, out=p)  # remove darkfield
            xp.divide(p, white, out=p)
            xp.log(p, out=p)  # take -log to linearize detector values
            xp.multiply(p, -1, out=p)  # sunny side up

    geometry = geom(settings, angles, corrections)

    vol_min, vol_max = suggest_volume_extent(geometry[0])
    projections = reflex.projs(projs_path, proj_ids)
    # vol = fdk(projections,
    #           geometry,
    #           (voxels_x, None, None),
    #           np.array(vol_min),
    #           np.array(vol_max),
    #           chunk_size=50,
    #           preproc_fn=_preproc)

    vol = sirt_experimental(projections,
                            geometry,
                            (voxels_x, None, None),
                            np.array(vol_min),
                            np.array(vol_max),
                            iters=50,
                            preproc_fn=_preproc)

    if plot_pyqtgraph:
        import pyqtgraph as pq
        pq.image(vol)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(vol[..., vol.shape[2] // 2])
        plt.show()


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
        voxels_x=200,
        proj_ids=np.arange(0, 1201, 5),
        plot_pyqtgraph=True,
        corrections=True)
