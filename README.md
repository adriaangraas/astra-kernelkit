# XRayCity
_ASTRA Toolbox experimental subproject._ 
This is an **unofficial and unstable proof-of-concept package** that implements the CUDA kernels from ASTRA Toolbox kernels on top of Cupy.

The package is mostly geared towards power users of the ASTRA Toolbox that need tuned performance or algorithmic development. 
 - from the integration with Cupy we can expect
    - support for streams so that we can dynamically offload/onload projections while the kernel is computing;
    - enable pinned host memory when possible to prevent page faults;
    - DLPack tensor compatibility with e.g. PyTorch and Chainer.
 -  from the Python high-level API we get some development benefits such as
    - debugging, in-between copying of memory 
    - easy plotting of algorithmic progress
    - testing frameworks are more friendly in Python
 - dynamic kernel compilation
   - remove features from the kernel before the compilation by templating the source code
   - no expensive axis swapping when order of the input sinogram or volume is given
   - easy mixed precision support
   - kernel optimization (Kerneltuner?)
 - easier algorithmic development
   - algorithms in pure Python lifts most of the two-language barrier
   - integration with cuBLAS, cuSPARSE and cuTENSOR
   - when ODL starts supporting CupyTensorSpace there is a neat integration that would not require GPU-CPU copying

It of course also has disadvantages, mainly being more difficult low-level integration with libraries (they should now
become Cython extensions to Cupy), perhaps the installation is more error-prone and it is a bit annoying to be
dependent on CuPy to timely update when CUDA updates are rolled out.

## Roadmap
#### PoC release
 - [ ] high level interface
    - [ ] geometry
 - [ ] ODL integration
 - [ ] cone beam 3D
 - [ ] constant memory (see https://github.com/cupy/cupy/pull/2510)
 - [ ] not-ODL examples 

#### 0.1 release
 - [ ] templated kernels
 - [ ] pinned host memory
 - [ ] volume/sinogram axis order by user
 - [ ] parallel beam 2D / 3D
 - [ ] issue #1
 - [ ] checks: axis, contiguous, dtypes

#### 0.2 release

#### 1.0 release
 - [ ] tests   
   - [ ] scaling, matching, contiguous
 - [ ] pip and Conda packaging
 - [ ] proper docs 
 - [ ] pitched/managed memory allocation 
 - [ ] support supersampling operations
 - [ ] correctly handle anisotropic pixel/voxel sizes
 
#### wishlist (not scheduled)
 - [ ] OpenCL backend support
 - [ ] kerneltuner 
 - [ ] mixed precision
 - [ ] issue #2
 - [ ] anisotropic voxel/pixel support

 
## Issues
#### #1 prevent copying 
Prevent copying the sinogram to GPU or volume to GPU if the operation 
after it is going to overwrite it anyway. Use some allocation algorithm
from CuPy.

#### #2 How to handle CuPy arrays in ODL?
The problem is that you don't want to switch to Numpy arrays internally
because this forces a CPU-GPU copy.
Probably not handled transparently: we need CupyTensorSpaces.

#### #3 Cast warnings when the user mixes host/device dtypes
Using higher precision memory is probably unnecessary slow and RAM consuming
when the user performs computations on the GPU.
 