# class Filter(Kernel):
#     ANGLES_PER_WEIGHT_BLOCK = 16
#     DET_BLOCK_U = 32
#     DET_BLOCK_V = 32
#
#     def __init__(self, *args, **kwargs):
#         raise DeprecationWarning("I'm not sure this works well.")
#         super().__init__('filter.cu', *args, **kwargs)
#
#     def _compile(self):
#         module = self.load_module(
#             det_block_u=self.DET_BLOCK_U,
#             det_block_v=self.DET_BLOCK_V,
#             angles_per_weight_block=self.ANGLES_PER_WEIGHT_BLOCK)
#         return module, {'preweight': module.get_function("preweight")}
#
#     def __call__(self,
#                  projections: list,
#                  geoms: list,
#                  filter: Any,
#                  short_scan: bool = False):
#         """
#         Reading material:
#             https://people.csail.mit.edu/bkph/courses/papers/Exact_Conebeam/Turbell_Thesis_FBP_2001.pdf
#
#         :param projections:
#         :param geoms:
#         :param filter:
#         :param short_scan:
#         :return:
#         """
#         for proj in projections:
#             if not isinstance(proj, cp.ndarray):
#                 raise TypeError("`projections` must be a CuPy ndarray.")
#
#         # NB: We don't support arbitrary cone_vec geometries here.
#         # Only those that are vertical sub-geometries
#         # (cf. CompositeGeometryManager) of regular cone geometries.
#         assert len(geoms) > 0
#
#         # TODO(Adriaan): assert geometry consistency
#         g0 = geoms[0]
#         nr_cols = g0.detector.cols
#         nr_rows = g0.detector.rows
#         nr_geoms = len(geoms)
#
#         # TODO(Adriaan): assert that geometry works well enough with FDK?
#         #  I guess this will do for now
#         if g0.u[2] != 0:
#             warnings.warn("Filter only works for geometries "
#                           "in the horizontal plane.")
#
#         # assuming U is in the XY plane, V is parallel to Z axis
#         det_cx, det_cy = (g0.detector_position
#                           + .5 * g0.u * g0.detector.width)[:2]
#         tube_origin = np.linalg.norm(g0.tube_position[:2])
#         det_origin = np.linalg.norm(np.array([det_cx, det_cy]))
#         det_cz = (g0.detector_position + .5 * g0.v[2] * g0.detector.height)[2]
#         z_shift = det_cz - g0.tube_position[2]
#
#         # TODO(ASTRA): FIXME: Sign/order
#         angles = [-np.arctan2(g.tube_position[0], g.tube_position[1]) + np.pi
#                   for g in geoms]
#
#         # The pre-weighting factor for a ray is the cosine of the angle between
#         # the central line and the ray.
#         module, funcs = self._compile()
#         blocks_u = int(np.ceil(nr_cols / self.DET_BLOCK_U))
#         blocks_v = int(np.ceil(nr_rows / self.DET_BLOCK_V))
#         blocks_angles = int(np.ceil(nr_geoms / self.ANGLES_PER_WEIGHT_BLOCK))
#         funcs['preweight'](
#             (blocks_u * blocks_v, blocks_angles),
#             (self.DET_BLOCK_U, self.ANGLES_PER_WEIGHT_BLOCK),
#             (cp.array([p.data.ptr for p in projections]),
#              0,
#              len(angles),
#              cp.float32(tube_origin),
#              cp.float32(det_origin),
#              cp.float32(z_shift),
#              cp.float32(g0.detector.pixel_width),
#              cp.float32(g0.detector.pixel_height),
#              nr_geoms,
#              nr_cols,
#              nr_rows))
#
#         if short_scan and len(geoms) > 1:
#             raise NotImplementedError("Parker weighting not implemented.")
#
#         filter(projections)