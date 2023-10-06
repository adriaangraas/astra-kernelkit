import numpy as np

class FanProjection(Kernel):
    """Fanbeam forward projection"""
    ANGLES_PER_BLOCK = 16
    DET_BLOCK_SIZE = 32
    BLOCK_SLICES = 64
    MAX_ANGLES = 25
    FAN_FP_FUNCTION = "fan_fp"

    def __init__(self, path: str = "../cuda/2d/fan_fp.cu"):
        super().__init__(path)

        # TODO(Adriaan): merge the two modules, like cone_fp
        self.horizontal = self.load_module(
            angles_per_block=self.ANGLES_PER_BLOCK,
            det_block_size=self.DET_BLOCK_SIZE,
            block_slices=self.BLOCK_SLICES,
            max_angles=self.MAX_ANGLES,
            mode_horizontal=True)

        self.vertical = self.load_module(
            angles_per_block=self.ANGLES_PER_BLOCK,
            det_block_size=self.DET_BLOCK_SIZE,
            block_slices=self.BLOCK_SLICES,
            max_angles=self.MAX_ANGLES,
            mode_horizontal=False)

    def __call__(self,
                 volume: astrapy.Volume,
                 sino: astrapy.Sinogram,
                 geometry: list[astrapy.Static2DGeometry],
                 rays_per_pixel: int = 1) -> astrapy.Sinogram:
        """Forward projection with fan geometry."""

        if isinstance(sino.measurement, np.ndarray):
            sino.measurement = cp.asarray(sino.measurement, dtype=self.FLOAT_DTYPE)
        elif isinstance(sino.measurement, cp.ndarray):
            if sino.measurement.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`projections` must be a ndarray.")

        if sino.ndim != 2:
            raise ValueError("`projections` must have exactly 2 dimensions.")

        if isinstance(volume.measurement, np.ndarray):
            volume.measurement = cp.asarray(volume.measurement, dtype=self.FLOAT_DTYPE)
        elif isinstance(volume.measurement, cp.ndarray):
            if volume.measurement.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`volume` must be a Numpy or CuPy ndarray.")

        if volume.ndim != 2:
            raise ValueError("`volume` must have exactly 2 dimensions.")

        if not volume.has_isotropic_voxels:
            raise NotImplementedError(
                f"`{self.__class__.__name__}` is not tested with anisotropic voxels yet.")

        # convert angles so that detector position is not the center point but the edge point
        # TODO: this should be abstracted away as an operation on a geometry
        nr_pixels = geometry[0].detector.nr_pixels
        # TODO: now t=0, we need a guarantee that the detector does not change
        # nr_pixels = sino.data.shape[1]

        vectors = []
        for geom in geometry:
            # find the starting point on the side of the detector
            # pixel_vector = np.multiply(geom.detector.pixel_width,
            #     astrapy.Flat1DDetector.AXIS_HORIZONTAL)
            # det_direction = geom.detector_rotation_matrix @ pixel_vector
            det_direction = geom.u * geom.detector.pixel_width

            left_detector_position = (
                geom.detector_position -
                det_direction * nr_pixels / 2)

            # convert to vector description
            vectors.append(
                [*left_detector_position,
                 *det_direction,
                 *geom.source_position])

        # upload angles to GPU
        # plot_geoms = {i: g for i, g in geometry.items() if i < 100 and i % 10 == 0}
        # astrapy.geom2d.plot(plot_geoms)

        # initialize TODO: what is the performance here, can we do this faster?
        sino.measurement.fill(0.)

        volume_texture = _cupy_copy_to_texture(volume.measurement)

        output_scale = volume.voxel_size[0]

        # chunk arrays in blocks of MAX_ANGLES
        for angle_start in range(0, len(vectors), self.MAX_ANGLES):
            angle_end = min(angle_start + self.MAX_ANGLES, len(vectors))
            self._call_chunk(volume_texture,
                             sino.measurement[angle_start:angle_end],
                             vectors[angle_start:angle_end],
                             rays_per_pixel,
                             output_scale)

        return sino

    def _call_chunk(self,
                    volume_texture: txt.TextureObject,
                    projections,
                    angles,
                    rays_per_pixel: int,
                    output_scale: float):
        assert len(projections) == len(angles)

        nr_pixels = projections.shape[1]
        angles /= output_scale

        angle_block_start, angle_block_end = 0, len(angles)
        # angle blocks, regions
        grid_x = (angle_block_end - angle_block_start
                  + self.ANGLES_PER_BLOCK - 1) // self.ANGLES_PER_BLOCK
        grid_y = (nr_pixels
                  + self.DET_BLOCK_SIZE - 1) // self.DET_BLOCK_SIZE

        # TODO use pinned host memory?
        # memory_pool = cp.cuda.PinnedMemoryPool()
        # cp.cuda.set_pinned_memory_allocator(memory_pool.malloc)

        # we have two streams, one for each kernel
        streams = [cp.cuda.stream.Stream()] * 2
        kernels = (self.horizontal.get_function(self.FAN_FP_FUNCTION),
                   self.vertical.get_function(self.FAN_FP_FUNCTION))
        modules = (self.horizontal, self.vertical)

        for stream, kernel, module in zip(streams, kernels, modules):
            _cupy_copy_to_constant(module, 'csrcX', angles[:, 4::6])
            _cupy_copy_to_constant(module, 'csrcY', angles[:, 5::6])
            _cupy_copy_to_constant(module, 'cdetSX', angles[:, 0::6])
            _cupy_copy_to_constant(module, 'cdetSY', angles[:, 1::6])
            _cupy_copy_to_constant(module, 'cdetUX', angles[:, 2::6])
            _cupy_copy_to_constant(module, 'cdetUY', angles[:, 3::6])

            # note that the cuda are executed asynchronously
            with stream:
                for start_slice in range(0, volume_texture.ResDesc.cuArr.width,
                                         self.BLOCK_SLICES):
                    kernel((grid_x, grid_y),
                           (self.DET_BLOCK_SIZE, self.ANGLES_PER_BLOCK),
                           (volume_texture,
                            projections,
                            # angles,
                            start_slice,
                            angle_block_start,
                            angle_block_end,
                            nr_pixels,
                            rays_per_pixel,
                            volume_texture.ResDesc.cuArr.width,
                            volume_texture.ResDesc.cuArr.height,
                            cp.float32(output_scale)))

        cp.cuda.Device().synchronize()


class FanBackprojection(Kernel):
    BLOCK_SLICES = 16
    BLOCK_SLICE_SIZE = 32
    ANGLES_PER_BLOCK = 16
    MAX_ANGLES = 2560
    FAN_BP_FUNCTION = "fan_bp"

    def __init__(self, path: str = "../cuda/2d/fan_bp.cu",
                 function="fan_bp"):
        super().__init__(path)
        self.module = self.load_module(
            block_slices=self.BLOCK_SLICES,
            block_slice_size=self.BLOCK_SLICE_SIZE,
            angles_per_block=self.ANGLES_PER_BLOCK,
            max_angles=self.MAX_ANGLES)

    def __call__(self,
                 volume: astrapy.Volume,
                 sino: astrapy.Sinogram,
                 geometry: list[astrapy.Static2DGeometry]
                 ) -> astrapy.Volume:
        """Backprojection with fan geometry."""

        if isinstance(sino.measurement, np.ndarray):
            sino.measurement = cp.asarray(sino.measurement, dtype=self.FLOAT_DTYPE)
        elif isinstance(sino.measurement, cp.ndarray):
            if sino.measurement.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`sino` must be a 2-dimensional ndarray.")

        if sino.measurement.ndim != 2:
            raise ValueError("`sino` must have exactly 2 dimensions.")

        if isinstance(volume.measurement, np.ndarray):
            volume.measurement = cp.asarray(volume.measurement, dtype=self.FLOAT_DTYPE)
        elif isinstance(volume.measurement, cp.ndarray):
            if volume.measurement.dtype not in self.SUPPORTED_DTYPES:
                raise NotImplementedError(
                    f"Currently there is only support for dtype={self.SUPPORTED_DTYPES}.")
        else:
            raise TypeError("`volume` must be a Numpy or CuPy ndarray.")

        if volume.ndim != 2:
            raise ValueError("`volume` must have exactly 2 dimensions.")

        if not volume.has_isotropic_voxels:
            # I think scaling will go wrong (i.e. ASTRA Toolbox cannot do non-square pixels as well)
            raise NotImplementedError(
                f"`{self.__class__.__name__}` is not tested with anisotropic voxels yet.")

        # initialize, TODO: can this be done faster/is this necessary here?
        volume.measurement.fill(0.)

        # TODO: circular geometry is assumed, we need to either assert
        #    that the geometry is circular here, or generalize to
        #    non-circular geometries
        # TODO: this should be abstracted away as an operation on a geometry
        converted_geometry = copy.deepcopy(geometry)
        output_scale = volume.voxel_size[0]

        # convert angles so that detector position is not the center point but the edge point
        # TODO: I should use decorator patterns here on geometries
        for ngeom, geom in zip(converted_geometry, geometry):
            ngeom.detector_position[:] = (
                geom.detector_position
                - geom.u * geom.detector.width / 2)

            # then shift to the center of the geom?
            dx = -(volume.extent_min[0] + volume.extent_max[0]) / 2
            dy = -(volume.extent_min[1] + volume.extent_max[1]) / 2
            ngeom.source_position[:] = ngeom.source_position + [dx, dy]
            ngeom.detector_position[:] = ngeom.detector_position + [dx, dy]

            # scale appropriately
            s = 1. / output_scale
            ngeom.source_position[:] = s * ngeom.source_position[:]
            ngeom.detector_position[:] = s * ngeom.detector_position[:]
            ngeom.detector.pixel_width = s * ngeom.detector.pixel_width

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(sino.data.get())
        # plt.show()
        # astrapy.geom2d.plot(converted_geometry)
        proj_texture = _cupy_copy_to_texture(sino.measurement)

        params = np.array(
            [a.to_list() for a in self._geoms2params(converted_geometry)])

        # TODO: refactor into a "chunking method", to share logic with fan_fp
        # chunk arrays in blocks of MAX_ANGLES
        for angle_start in range(0, len(geometry), self.MAX_ANGLES):
            angle_end = min(angle_start + self.MAX_ANGLES, len(geometry))
            self._call_chunk(volume.measurement,
                             proj_texture,
                             params[angle_start:angle_end],
                             output_scale)

        # angle_extent = self.geometry.motion_partition.extent
        # num_angles = self.geometry.motion_partition.shape
        # scaling_factor = (angle_extent / num_angles).prod()

        # TODO: nasty stuff, what if I don't have equidistant angles?
        #   I took this from ODL, but I'm not sure if they tested for arbitrary angles
        output_scale = 2 * np.pi / 360  # TODO

        # Correct in case of non-weighted spaces
        # proj_extent = float(self.proj_space.partition.extent.prod())
        # proj_size = float(self.proj_space.partition.size)

        # proj_weighting = proj_extent / proj_size
        # scaling_factor *= (self.proj_space.weighting.const /
        #                    proj_weighting)
        # scaling_factor /= (self.reco_space.weighting.const /
        #                    self.reco_space.cell_volume)

        output_scale /= float(volume.voxel_volume)
        output_scale *= float(sino.pixel_volume)

        volume.measurement *= output_scale

        return volume

    def _call_chunk(self, volume, proj_texture, params, output_scale: float):
        assert (len(params) < self.MAX_ANGLES)
        grid_x = (volume.shape[0]
                  + self.BLOCK_SLICES - 1) // self.BLOCK_SLICES
        grid_y = (volume.shape[1]
                  + self.BLOCK_SLICE_SIZE - 1) // self.BLOCK_SLICE_SIZE

        with cp.cuda.stream.Stream():
            _cupy_copy_to_constant(self.module, 'params', params.flatten())
            for start_angle in range(0, len(params), self.ANGLES_PER_BLOCK):
                self.module.get_function(self.FAN_BP_FUNCTION)(
                    (grid_x, grid_y),
                    (self.BLOCK_SLICES, self.BLOCK_SLICE_SIZE),
                    (proj_texture,
                     volume,
                     start_angle,
                     len(params),
                     volume.shape[0],
                     volume.shape[1],
                     cp.float32(output_scale)))

        cp.cuda.Device().synchronize()

    def _geoms2params(self, geoms):
        """We need three values in the kernel:
         - projected coordinates of pixels on the detector:
           || x (s-d) || + ||s d|| / || u (s-x) ||
         - ray density weighting factor for the adjoint
           || u (s-d) || / ( |u| * || u (s-x) || )
         - fan-beam FBP weighting factor
           ( || u s || / || u (s-x) || ) ^ 2
        """

        class _Param:
            """Inner class to support kernel parameter computations."""

            def __init__(self, num_c, num_x, num_y, den_c, den_x, den_y):
                self.num_c = num_c
                self.num_x = num_x
                self.num_y = num_y
                self.den_c = den_c
                self.den_x = den_x
                self.den_y = den_y

            def to_list(self):
                return [self.num_c, self.num_x, self.num_y,
                        self.den_c, self.den_x, self.den_y]

        fan_params = []
        for geom in geoms:
            src = geom.source_position
            detector = geom.detector_position
            pixel_vector = geom.u * geom.detector.pixel_width

            # goal: 1/fDen = || u s || / || u (s-x) ||
            # fDen = || u (s-x) || / || u s ||
            # i.e., scale = 1 / || u s ||
            # scale = 1.0 / det(u, s)
            scale = (
                np.linalg.norm(pixel_vector) /
                np.linalg.det((pixel_vector, src - detector))
            )

            p = _Param(
                num_c=scale * np.linalg.det((src, detector)),
                num_x=scale * (src - detector)[1],
                num_y=-scale * (src - detector)[0],
                den_c=scale * np.linalg.det((pixel_vector, src)),
                den_x=scale * pixel_vector[1],
                den_y=-scale * pixel_vector[0]
            )
            fan_params.append(p)

        return fan_params

