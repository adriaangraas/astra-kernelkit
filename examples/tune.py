from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from kernel_tuner.core import CompilationInstance
import kernel_tuner.devices.cupy as kt_cupy
from kernel_tuner.device import DeviceInterface
from kernel_tuner.interface import tune
from kernel_tuner.runners.sequential import SequentialRunner

from astrapy import Detector, Geometry, fp, rotate, ConeBackprojection, ConeProjection
from astrapy.kernel import _to_texture


class RayveInstance(CompilationInstance):
    """Class that groups the Cupy functions on maintains state about the device"""

    def __init__(self, func, threads, grid):
        super().__init__(threads, grid)
        self.func = func

    def run_kernel(self, arguments, stream=None):
        """runs the CUDA kernel passed as 'func'

        :param func: A cupy kernel compiled for this specific kernel configuration
        :type func: cupy.RawKernel

        :param gpu_args: A list of arguments to the kernel, order should match the
            order in the code. Allowed values are either variables in global memory
            or single values passed by value.
        :type gpu_args: list( cupy.ndarray, numpy.int32, ...)

        :param threads: A tuple listing the number of threads in each dimension of
            the thread block
        :type threads: tuple(int, int, int)

        :param grid: A tuple listing the number of thread blocks in each dimension
            of the grid
        :type grid: tuple(int, int)
        """
        self.func(**arguments)


class RayveInterface(DeviceInterface):
    def __init__(self, kernel_options, compiler_options, quiet=False, device=0,
                 **kwargs):
        super().__init__(kernel_options)
        self.dev = dev = cp.cuda.Device(device).__enter__()

        #inspect device properties
        self.devprops = dev.attributes
        self.cc = dev.compute_capability
        env = dict()
        cupy_info = str(cp._cupyx.get_runtime_info()).split("\n")[:-1]
        info_dict = {s.split(":")[0].strip(): s.split(":")[1].strip() for s in
                     cupy_info}
        env["device_name"] = info_dict[f'Device {device} Name']
        env["cuda_version"] = cp.cuda.runtime.driverGetVersion()
        env["compute_capability"] = self.cc
        env["compiler_options"] = compiler_options
        env["device_properties"] = self.devprops
        self._env = env

        # create a stream and events
        self.stream = cp.cuda.Stream()
        self.start = cp.cuda.Event()
        self.end = cp.cuda.Event()

    @property
    def name(self):
        return self._env["device_name"]

    def upload_arguments(self, arguments):
        return []

    def get_environment(self):
        pass

    def units(self):
        return {'ms'}

    def get_environment(self):
        return self._env

    def max_threads(self):
        return self.devprops['MaxThreadsPerBlock']

    def check_kernel_output(self, func, gpu_args, instance, answer, atol, verify, verbose):
        pass

    def benchmark(self, instance, arguments, iterations, observers, verbose=True):
        return kt_cupy._benchmark(self.dev, instance, arguments, self.start, self.end, self.stream,
                   iterations, observers, verbose)


class RayveBpInterface(RayveInterface):
    def compile_kernel(self, tune_params, verbose=True) -> CompilationInstance:
        grid, threads = self.grid_and_threads(tune_params)
        voxels_per_block = (tune_params['block_size_x'],
                            tune_params['block_size_y'],
                            tune_params['block_size_z'])
        bpkern = ConeBackprojection(
            voxels_per_block=voxels_per_block,
        )
        bpkern.compile()
        compilation = RayveInstance(bpkern, threads, grid)
        return compilation


class RayveFpInterface(RayveInterface):
    def compile_kernel(self, tune_params, verbose=True) -> CompilationInstance:
        grid, threads = self.grid_and_threads(tune_params)
        fpkern = ConeProjection(
            slices_per_thread=tune_params['slices_per_thread'],
            cols_per_thread=tune_params['cols_per_thread'],
            rows_per_block=tune_params['rows_per_block'])
        fpkern.compile(719)
        compilation = RayveInstance(fpkern, threads, grid)
        return compilation


geom_t0 = Geometry(
    [-10, 0., 0.],
    [20, 0., 0.],
    [0, 1, 0],
    [0, 0, 1],
    Detector(99, 141, .01, .01))
geom_t0 = rotate(geom_t0, yaw=.5 * np.pi, roll=.2 * np.pi, pitch=.1 * np.pi)

angles = np.linspace(0, 2 * np.pi, 719)
geoms = [rotate(geom_t0, yaw=a) for a in angles]
vol_min, vol_max = [-.2] * 3, [.2] * 3

# cube with random voxels
vol = cp.zeros([200] * 3, cp.float32)
vol[25:75, 25:75, 25:75] = cp.random.random(tuple([50] * 3))
vol[35:65, 35:65, 35:65] += cp.random.random(tuple([30] * 3))
vol[45:55, 45:55, 45:55] += 1.


kernel_options = OrderedDict({
    'grid_div_x': None, 'grid_div_y': None, 'grid_div_z': None,
    'block_size_names': None,
    'problem_size': (len(angles), geom_t0.detector.rows, geom_t0.detector.cols)
})
compiler_options = {}

projs = [cp.zeros((g.detector.rows, g.detector.cols), dtype=np.float32) for g in geoms]
dev = RayveFpInterface(kernel_options, compiler_options)
arguments = {
    'volume_texture': _to_texture(vol),
    'volume_extent_min': vol_min,
    'volume_extent_max': vol_max,
    'geometries': geoms,
    'projections': projs
}
runner = SequentialRunner(dev, [kt_cupy.CupyRuntimeObserver(dev)], 7, arguments, kernel_options)
tune_params = {
    'slices_per_thread': [4, 8, 16, 32, 64],
    'cols_per_thread': [4, 8, 16, 32, 64],
    'rows_per_block': [16, 32, 64]}

from time import perf_counter

t = perf_counter()
tune(runner, t, tune_params)





#
# # forward project
# projs = fp(vol, geoms, vol_min, vol_max)
# # projs_gpu = [cp.asarray(p) for p in projs]
# projs_txt = [_to_texture(p) for p in projs]
# vol2 = cp.zeros_like(vol)
#
#
# kernel_options = OrderedDict({
#     'grid_div_x': None, 'grid_div_y': None, 'grid_div_z': None,
#     'block_size_names': None,
#     'problem_size': vol2.shape
# })
# compiler_options = {}
#
# dev = RayveInterface(kernel_options, compiler_options)
# arguments = {
#     'projections_textures': projs_txt,
#     'geometries': geoms,
#     'volume': vol2,
#     'volume_extent_min': vol_min,
#     'volume_extent_max': vol_max
#
# }
# runner = SequentialRunner(dev, [kt_cupy.CupyRuntimeObserver(dev)], 7, arguments, kernel_options)
# tune_params = {
#     'block_size_x': [2, 4, 8, 16, 32, 64],
#     'block_size_y': [2, 4, 8, 16, 32, 64],
#     'block_size_z': [4, 6, 8, 10]}
#
# from time import perf_counter
#
# t = perf_counter()
# tune(runner, t, tune_params)