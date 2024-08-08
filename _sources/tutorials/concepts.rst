.. _concepts:

==============================================
KernelKit low-level concepts explained in code
==============================================

KernelKit's main value point is that is it allows users to modify any of the nitty-gritty details of reconstruction -- all the way from the GPU level to the level of X-ray backprojection and reprojection. To do so, KernelKit splits reconstruction into CUDA code, Kernel classes, and Projector classes. Projectors are a high-level building block that can be used in algorithms.

In this bird's-eye view, we will go from the low-level (CUDA sources) to the high-level (Projectors). We will explain what KernelKit does at each level with a simplified example for 2D fanbeam backprojection. The code supporting this tutorial is available at :github:`example/tutorials/kernelkit_concepts.ipynb`.


1. CUDA sources
===============

The core computations of backprojection and reprojection are executed in CUDA kernels, which are comparatively small C++ functions that execute on NVIDIA GPUs using the `CUDA platform <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>`_.The following is a simplified example:

.. code-block:: C++

        __global__ void fan_bp(
            float * volume, float * sinogram, float * angles,
            int nr_angles, int nr_voxels_x, int nr_voxels_y, int nr_pixels,
            float src_obj_dist, float obj_det_dist
        ) {
            // X, Y are the voxel's center coordinates of this thread
            unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
            unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
            float X = float(x) - nr_voxels_x / 2.0 + 0.5;
            float Y = float(y) - nr_voxels_y / 2.0 + 0.5;

            // ignore threads outside the volume
            if (x >= nr_voxels_x || y >= nr_voxels_y)
                return;

            for (int i = 0; i < nr_angles; i++) {
                // project the voxel (X, Y) onto a coordinate U on the detector line
                // (full source in Jupyter notebook)
                float U = xray_project(X, Y, angles[i], src_obj_dist, obj_det_dist);
 
                // linearly interpolate the value at U from sinogram gridpoints
                volume[y * nr_voxels_x + x] += linearly_interpolate(sinogram, i, nr_pixels, U);
            }
        }

In this CUDA function, aimed at backprojection for circular fanbeam CT, arrays :python:`float * volume` and :python:`float * sinogram` are passed as arguments, as well as some geometry parameters of the 2D reconstruction problem. The arrays reside on the GPU in global memory.

Each parallel thread of this CUDA kernel loops over all angles in the sinogram to compute the value at a single voxel :math:`(X, Y)` in the 2D volume. This uses a voxel-driven backprojection approach. The reader is referred to :footcite:t:`hansen_2021` (Chapter 9) for an explanation of X-ray projection.

Importantly, CUDA kernels enable a matrix-free approach of computing the sinogram
from the volume, or vice versa. A projection matrix is never explicitly formed in memory thanks to its coefficients being computed on-the-fly in the CUDA function.


2. :python:`kernelkit.Kernel`
=============================

To launch CUDA kernels from Python, like the one above, KernelKit provides :python:`Kernel` classes. These are thin wrappers that use :cupy:`cupy.RawModule` for compilation of the CUDA source code, as well as argument passing. The following example :python:`FanBp` provides this for the fanbeam example above.

.. code-block:: Python

        class FanBp(kernelkit.kernel.BaseKernel):
            """Fanbeam backprojection for circular geometry."""

            def __init__(self):
                cuda_source = Path(__file__).with_name("fan_bp.cu").read_text()
                super().__init__(cuda_source)

            def compile(self):
                self._compile(("fan_bp",))

            def __call__(self, volume: cp.ndarray, sinogram: cp.ndarray,
                         angles: cp.ndarray, src_dist: float, det_dist: float):
                # here we retrieve a function using CuPy 
                fan_bp = self._module.get_function("fan_bp")

                # calling the module launches the kernel from the code example above
                fan_bp(
                    (32, 32),                                                # CUDA threads
                    (volume.shape[1] // 32 + 1, volume.shape[0] // 32 + 1),  # CUDA blocks
                    (                                                        # arguments
                        volume, sinogram, angles,
                        len(angles), *volume.T.shape, sinogram.shape[1],  # width, height, det_count
                        cp.float32(det_spacing), cp.float32(src_dist), cp.float32(det_dist)
                    )
                )


The :python:`__call__(...)` enables us to invoke :python:`FanBp` similarly to a function. In this case, it will launch the CUDA kernel with 32-by-32 sized thread blocks. Each thread block thus processes a small part of the :python:`input` volume.

Although unpractical, one would already be able to use a kernel directly in a backprojection algorithm. Suppose we would have a data file `sinogram.npy`, now :python:`FanBp` can be used like so:

.. code-block:: Python

        output = cp.zeros((100, 100), dtype=cp.float32) 
        data   = cp.asarray(np.load('sinogram.npy'), dtype=cp.float32)
        fan_bp = FanBp()
        fan_bp.compile()
        fan_bp(output, data, [...])  # geometry omitted in this example
        # the kernel has written in `output` 


Practically, kernel classes like the above need to be more extensive, for instance to check the validity of the arguments, or to prepare the C++ code before compilation.


3. :python:`kernelkit.Projector`
================================

:python:`Projector` objects are a KernelKit concept to streamline computing reconstructions using kernels. Often, additional memory management and geometry computations are required around the execution of a kernel, which can be conveniently handled in a projector object. A projector can furthermore help recycle allocated memory so that repeated execution of the same kernel are more efficient.

In the following simplified example, we introduce a `FanbeamBackProjector`. This projector splits the sinogram into chunks before uploading them to GPU. This enables reconstruction with large quantities of sinogram data without exceeding the GPU memory capacity.

.. code-block:: Python

    class FanbeamBackProjector(kernelkit.BaseProjector):
        """Chunking backprojection for a circular 2D geometry."""

        def __init__(self, nr_chunks: int):
            super(FanbeamProjector, self).__init__()
            self._kernel = FanBp()
            self._kernel.compile()
            assert nr_chunks > 0
            self._nr_chunks = nr_chunks

        # [getters and setters are removed for brevity]

        def __call__(self):
            # split the projection indices and the geometry angles in chunks
            inds = np.array_split(np.arange(len(self._geom['angles'])),
                                  self._nr_chunks)
            angles = np.array_split(self._geom['angles'],
                                    self._nr_chunks)

            for inds, angles in zip(inds, angles):  # chunk-by-chunk
                print(f"Uploading and backprojecting [{inds[0]}, {inds[-1]})...")

                # call the `FanBp()` kernel that we wrote in section 2.
                self._kernel(
                    self.volume,
                    cp.asarray(self.projections[inds]),  # upload
                    angles,                              # angular subset
                    self._geom['det_spacing'],
                    self._geom['src_obj_dist'],
                    self._geom['obj_det_dist'])


In the :python:`__call__()` of this projector, :python:`self.volume` is retained on the
GPU, while :python:`cp.asarray(self.projections[inds])` uploads a new angular chunk of the
sinogram data. The projector can be used in the following example:

.. code-block:: Python

    # upload and backproject in 7 chunks
    bp = FanbeamBackProjector(nr_chunks=7)
    bp.volume = volume
    bp.projections = projs
    bp.projection_geometry = { ... }
    bp()  # invokes __call__
    # the kernel has written in `output`

Further reading
---------------

We've explained the steps between CUDA kernel that computes the backprojection and the Python X-ray transform algorithm.

