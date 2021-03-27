from typing import Any

import cupy as cp
import numpy as np
import reflex
from reflex import centralize

from astrapy import bp
from astrapy.geom3d import AstraStatic3DGeometry, Flat2DDetector


class Reconstruction:
    def __init__(self,
                 path: str,
                 settings_path: str = None,  # defaults to path
                 proj_range: range = None,
                 corrections=None,
                 verbose=True,
                 ):
        self._path = path
        if settings_path is None:
            settings_path = path

        self._settings_path = path
        self._settings = reflex.Settings.from_path(settings_path)
        self._proj_range = proj_range
        self.verbose = verbose
        self.corrections = corrections

    @property
    def settings(self):
        return self._settings

    def load_sinogram(self,
                      projs_path: str = None,
                      darks_path: str = None,
                      flats_path: str = None):
        if projs_path is None:
            projs_path = self._path
        if darks_path is None:
            darks_path = self._path
        if flats_path is None:
            flats_path = self._path

        # Preprocess and convert into C-order float32 ASTRA projection data
        return (reflex.projs(projs_path, self._proj_range),
                reflex.darks(darks_path),
                reflex.flats(flats_path))

    def geom(self, angles, corrections: Any = True, verbose=None):
        if verbose is None:
            verbose = self.verbose

        motor_geom = reflex.motor_geometry(self.settings,
                                           corrections,
                                           verbose=verbose)
        geom = reflex.circular_geometry(self._settings,
                                        initial_geom=motor_geom,
                                        angles=angles)

        if self._proj_range is not None:
            if len(geom) != len(self._proj_range):
                raise ValueError("`proj_range` has to be equal to the number "
                                 "of geometry angles. Please use `angles` to "
                                 "select which angles belong to these projections.")

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
                pixel_height=g.detector.pixel_height
            )
            # _swp = lambda x: np.array([x[2], x[0], x[1]])
            _swp = lambda x: x
            geom = AstraStatic3DGeometry(
                tube_pos=_swp(g.tube_position),
                det_pos=_swp(g.detector_position),
                u_unit=_swp(hv),
                v_unit=_swp(vv),
                detector=det)
            geoms.append(geom)

        return geoms

    def detector(self):
        return reflex.SquarePixelDetector(self._settings.det_binned_rows(),
                                          self._settings.det_binned_cols(),
                                          self._settings.binned_pixel_size())

    def _compute_volume_dimensions(self, nr_voxels_x):
        # Create a detector (needed for the geometry)
        det = self.detector()

        # voxel width according to physical pixel size
        voxel_width, voxel_height = self._settings.suggested_voxel_size()

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

    def backward(self, projections, geometry,
                 algo='FDK',
                 voxels_x=None,
                 iters=200,
                 fpreproc=None):
        voxels_x, voxels_z, w, h = self._compute_volume_dimensions(voxels_x)
        volume_shape = (voxels_x, voxels_x, voxels_z)
        volume_ext_min, volume_ext_max = (-w, -w, -h), (w, w, h)

        print("Algorithm starts...")
        algo = algo.lower()
        if algo == 'fdk':
            vol = bp(projections,
                     geometry,
                     volume_shape,
                     volume_ext_min,
                     volume_ext_max,
                     chunk_size=70,
                     fpreproc=fpreproc,
                     filter='ram-lak')
        else:
            raise ValueError("Algorithm value incorrect.")

        return vol

    # @staticmethod
    # def forward(volume_id, volume_geom, projection_geom,
    #             returnData=False):
    #     return cuda.cone_fp(volume_id,
    #                            projection_geom,
    #                            volume_geom,
    #                            returnData=returnData)


def reconstruct(
    settings_path,
    projs_path,
    darks_path,
    flats_path,
    voxels_x,
    proj_range,
    angles=None,
    verbose=True,
    plot_pyqtgraph=False,
    reconstruction_type=Reconstruction,
    corrections: Any = True,
):
    rec = reconstruction_type(
        path=projs_path,
        settings_path=settings_path,
        proj_range=proj_range,
        verbose=verbose
    )

    if angles is None:
        angles = np.linspace(0, 2 * np.pi, len(proj_range), endpoint=False)

    projections, darks, whites = rec.load_sinogram(darks_path=darks_path,
                                                  flats_path=flats_path)
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
        # flatfield the projections
        cp.divide(projs, white, out=projs)
        cp.log(projs, out=projs)  # take -log to linearize detector values
        return cp.multiply(projs, -1, out=projs)  # sunny side up

    geometry = rec.geom(angles, corrections)
    vol_cpu = rec.backward(
        projections,
        geometry,
        voxels_x=voxels_x,
        fpreproc=_preproc_projs)

    if plot_pyqtgraph:
        import pyqtgraph as pq
        pq.image(vol_cpu)

    print(vol_cpu.shape)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(vol_cpu[..., 50])
    plt.show()

    return vol_cpu


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    path = "/home/adriaan/data/pink_plant"
    # corrections = reflex.motor.Corrections()
    # corrections.tra_det_correction = 24.
    # corrections.tra_obj_correction = -.6
    rec = reconstruct(
        settings_path=path,
        projs_path=path,
        darks_path=path,
        flats_path=path,
        voxels_x=400,
        # proj_range=range(0, 1201, 16),
        # angles=np.arange(0, 2 * np.pi, 2 * np.pi / 1201 * 16),
        proj_range=range(0, 1201),
        angles=np.arange(0, 2 * np.pi, 2 * np.pi / 1201),
        verbose=True,
        plot_pyqtgraph=True,
        corrections=True,
    )
