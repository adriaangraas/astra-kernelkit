import cupy as cp
import numpy as np

from kernelkit.kernel import copy_to_texture


def test_to_texture():
    kernel = cp.RawKernel(r'''
    extern "C" __global__
    void txt_test(texture<float, 3, cudaReadModeElementType> txt,
                  float * x2) {
        int tid = threadIdx.x * blockDim.y * blockDim.z
                + threadIdx.y * blockDim.z
                + threadIdx.z;
        // tex3D(..., z, y, x)
        x2[tid] = tex3D(txt, threadIdx.z + .5f, threadIdx.y + .5f, threadIdx.x + .5f); 
    }
    ''', 'txt_test')

    # 5 voxels x, 3 voxels y, 2 slices (voxels z)
    # slowest index is the x direction, fastest index is the slice
    x1 = cp.arange(5 * 3 * 2, dtype=cp.float32).reshape(5, 3, 2)
    # x1[..., 1] = 42.
    print(x1)

    txt = copy_to_texture(x1)
    print(x1.shape)  # still (5, 3, 2), good
    x2 = cp.zeros_like(x1, dtype=cp.float32)  # expect 5, 3, 2
    kernel((1, 1, 1),
           (5, 3, 2),
           (txt, x2))
    print(x2.get())

    assert x2.shape == x1.shape
    np.testing.assert_array_almost_equal(x2.get(), x1.get())


def test_compilation():
