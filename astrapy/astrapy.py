import numpy as np
import cupy as cp


class StaticGeometry:
    def __init__(self, det_x, det_y, det_u_x, det_u_y, tube_x, tube_y):
        self.det_x = det_x
        self.det_y = det_y

        self.det_u_x = det_u_x
        self.det_u_y = det_u_y

        self.tube_x = tube_x
        self.tube_y = tube_y

    # @todo is there a nicer way?
    def to_list(self):
        return [self.det_x, self.det_y, self.det_u_x, self.det_u_y, self.tube_x, self.tube_y]

    def __repr__(self):
        return f"Det={self.det_x};{self.det_y};u={self.det_u_x};{self.det_u_y} Tube={self.tube_x};{self.tube_y}"


class Volume:
    """n-dimensional reconstruction volume.

    I would love to use the ODL functionality here, but it would yield a direct ODL dependency. If you plan on using
    ODL, please see `astrapy.odlmod.tomo.backends.astrapy.DiscreteLpVolumeAdapter` instead.
    """

    def __init__(self, data, extent_min: tuple, extent_max: tuple, axis_order: tuple = None):
        self.data = data

        if len(extent_min) != data.ndim or len(extent_max) != data.ndim:
            raise ValueError("`extent_min` and `extent_max` require the same dimensions as the data.")

        if axis_order is not None:
            # @todo check if correct
            pass
        else:
            axis_order = tuple(range(data.ndim))

        self.extent_min = extent_min
        self.extent_max = extent_max
        self.axis_order = axis_order

    @property
    def data_numpy(self) -> np.ndarray:
        """Guarantees a Numpy array
        Use if the caller expects Numpy or needs explicit CPU operations.
        Might invoke a data transfer."""
        if isinstance(self._data, cp.ndarray):
            return cp.asnumpy(self._data)

        return self._data

    @property
    def data(self):
        """Returns either a Numpy or Cupy implementation, depending on the current internal state.
        Does not induce a data transfer between GPU/CPU.
        Use if you don't care where the array lives at the moment."""
        return self._data

    @data.setter
    def data(self, value):
        if not (isinstance(value, np.ndarray) or isinstance(value, cp.ndarray)):
            raise TypeError("`data` has to be a ndarray.")

        self._data = value

    @property
    def data_cupy(self) -> cp.ndarray:
        """Guarantees a Cupy implementation
        Use if you plan on explicit GPU or Cupy operations.
        Might invoke a data transfer."""
        return self._data

    @property
    def ndim(self) -> int:
        """The dimensions of the data array."""
        return self.data.ndim

    @property
    def voxel_size(self) -> tuple:
        """The physical size of a voxel."""
        n = np.array(self.data.shape)
        dists = np.array(self.extent_max) - np.array(self.extent_min)
        shp = list(dists / n)
        return tuple(shp)

    @property
    def has_isotropic_voxels(self) -> bool:
        """Check if a voxel has the same length in each direction."""
        shp = np.array(self.voxel_size)
        return np.all(shp - shp.flat[0] == 0.)

    @property
    def voxel_volume(self) -> float:
        return float(np.prod(self.voxel_size))


class Sinogram:
    """Wrapper for measured projection data for some times/angles

    The wrapper
     - adds some physicals dimensions that are required for good reconstructions;
     - provides Numpy access to the data via `data`:
     - allows the caller to obtain the data without memory transfer from the GPU, with `data_cupy`.
    """

    def __init__(self, data, extent_min: tuple, extent_max: tuple, axis_order: tuple = None):
        """Initialize a Sinogram data array

        :param data:
        :param extent_min:
            Physical extent of the detector
        :param extent_max:
            Physical extent of the detector
        :param axis_order:
            [0,1,2]
        """
        if not (isinstance(data, np.ndarray) or isinstance(data, cp.ndarray)):
            raise TypeError("`data` has to be a Numpy or Cupy ndarray.")

        if len(extent_min) != data.ndim - 1 or len(extent_max) != data.ndim - 1:
            raise ValueError("`extent_min` and `extent_max` require the same dimensions as the data - 1.")

        if axis_order != [0, 1, 2]:
            # @todo check if correct
            pass
        else:
            axis_order = tuple(range(data.ndim))

        self._data = data
        self.extent_min = extent_min
        self.extent_max = extent_max
        self.axis_order = axis_order

    @property
    def data_numpy(self) -> np.ndarray:
        """Guarantees a Numpy array
        Use if the caller expects Numpy or needs explicit CPU operations.
        Might invoke a data transfer."""
        if isinstance(self._data, cp.ndarray):
            return cp.asnumpy(self._data)

        return self._data

    @property
    def data(self):
        """Guarantees an array implementing __array_function__
        Returns either a Numpy or Cupy implementation, depending on the current internal state.
        Does not induce a data transfer between GPU/CPU.
        Use if you don't care where the array lives at the moment."""
        return self._data

    @data.setter
    def data(self, value):
        if not (isinstance(value, np.ndarray) or isinstance(value, cp.ndarray)):
            raise TypeError("`data` has to be a ndarray.")

        self._data = value

    @property
    def data_cupy(self) -> cp.ndarray:
        """Guarantees a Cupy implementation
        Use if you plan on explicit GPU or Cupy operations.
        Might invoke a data transfer."""
        return self._data

    @property
    def ndim(self) -> int:
        """The dimensions of the data array."""
        return self.data.ndim

    @property
    def pixel_size(self) -> tuple:
        """The physical size of a pixel."""
        # @todo consider ordering (now assuming angular space is first dim)
        n = np.array(self.data.shape)
        dists = np.array(self.extent_max) - np.array(self.extent_min)
        shp = list(dists / n)
        shp.pop()
        return tuple(shp)

    @property
    def has_isotropic_pixels(self) -> bool:
        """Check if a pixel has the same length in each direction."""
        # @todo consider ordering (now assuming angular space is first dim)
        shp = np.array(self.pixel_size[1:])
        return np.all(shp - shp.flat[0] == 0.)

    @property
    def pixel_volume(self) -> float:
        return float(np.prod(self.pixel_size))


def empty_gpu(shape, dtype=None, order=None):
    """An implementation-unaware array creation helper.
    Leaves the choice of array initialization up to ASTRApy."""
    return cp.empty(shape, dtype, order)
