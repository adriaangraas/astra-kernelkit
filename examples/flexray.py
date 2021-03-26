from typing import Any

import numpy as np
import reflex
from reflex import centralize, prepare_projs

from astrapy import bp
from astrapy.geom3d import AstraStatic3DGeometry, Flat2DDetector


class Reconstruction:
    def __init__(self,
                 path: str,
                 settings_path: str = None,  # defaults to path
                 proj_range: range = None,
                 prepf=prepare_projs,
                 corrections=None,
                 det_subsampling=1,
                 verbose=True,
                 ):
        self._path = path

        if settings_path is None:
            settings_path = path
        self._settings_path = path

        self._settings = reflex.Settings.from_path(settings_path)
        self._proj_range = proj_range

        self.prepf = prepf
        self.verbose = verbose
        self.corrections = corrections
        self.det_subsampling = det_subsampling

    @property
    def settings(self):
        return self._settings

    def load_sinogram(self,
                      projs_path: str = None,
                      darks_path: str = None,
                      flats_path: str = None,
                      ):
        if projs_path is None:
            projs_path = self._path

        if darks_path is None:
            darks_path = self._path

        if flats_path is None:
            flats_path = self._path

        # Preprocess and convert into C-order float32 ASTRA projection data
        return self.prepf(
            reflex.projs(projs_path, self._proj_range),
            reflex.darks(darks_path),
            reflex.flats(flats_path),
            downsampling=self.det_subsampling,
            verbose=self.verbose,
            transpose=None)

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
        # reflex.dynamic_geom_to_astra_vectors()
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
            _swp = lambda x: np.array([x[2], x[0], x[1]])

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

        # compensate for detector subsampling
        if self.det_subsampling != 1:
            # TODO(Adriaan): I have to check this after fixing a bug
            raise NotImplementedError()
            voxel_width /= self.det_subsampling
            voxel_height /= self.det_subsampling

        # compute total volume size based on the rounded voxel size
        width_from_center = voxel_width * nr_voxels_x / 2
        height_from_center = voxel_height * nr_voxels_z / 2

        return nr_voxels_x, nr_voxels_z, width_from_center, height_from_center

    def backward(self, projections, geometry,
                 algo='FDK', voxels_x=None, iters=200):
        voxels_x, voxels_z, w, h = self._compute_volume_dimensions(voxels_x)
        volume_shape = (voxels_x, voxels_x, voxels_z)
        volume_ext_min, volume_ext_max = (-w, -w, -h), (w, w, h)

        print("Algorithm starts...")
        algo = algo.lower()
        if algo == 'fdk':
            vol = bp(projections, geometry,
                     volume_shape, volume_ext_min, volume_ext_max,
                     chunk_size=50)
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

    projections = rec.load_sinogram(darks_path=darks_path,
                                    flats_path=flats_path)

    # plt.figure()
    # plt.imshow(projections[0])
    # plt.show()
    # projections[:] = 1.

    geometry = rec.geom(angles, corrections)
    vol_cpu = rec.backward(
        projections,
        geometry, voxels_x=voxels_x)

    if plot_pyqtgraph:
        import pyqtgraph as pq
        pq.image(vol_cpu)

    print(vol_cpu.shape)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(vol_cpu[..., 50])
    plt.show()

    return vol_cpu


# def reconstruct_odl():
#     import odl
#     dpart = detector_partition_3d(recon_width_length, recon_height_length,
#                                   DETECTOR_PIXEL_WIDTH,
#                                   DETECTOR_PIXEL_HEIGHT)
#     geometry = odl.tomo.ConeBeamGeometry(apart, dpart, SOURCE_RADIUS,
#                                          DETECTOR_RADIUS)
#
#     reco_space_3d = odl.uniform_discr(
#         min_pt=[-L, -L, -H],
#         max_pt=[L, L, H],
#         shape=[n, n, m], dtype=np.float32)
#     xray_transform = odl.tomo.RayTransform(reco_space_3d, geometry,
#                                            impl=MyImpl)

if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    path = "/home/adriaan/data/star_fruit"
    corrections = reflex.motor.Corrections()
    corrections.tra_det_correction = 24.
    corrections.tra_obj_correction = -.6
    rec = reconstruct(
        settings_path=path,
        projs_path=path,
        darks_path=path,
        flats_path=path,
        voxels_x=700,
        # proj_range=range(0, 1201),
        # angles=np.arange(0, 2 * np.pi, 2 * np.pi / 1201),
        proj_range=range(0, 1201),
        angles=np.arange(0, 2 * np.pi, 2 * np.pi / 1201),
        verbose=True,
        plot_pyqtgraph=True,
        corrections=corrections,
    )

    # pq.image(rec)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.show()
