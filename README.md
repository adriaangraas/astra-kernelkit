## The ASTRA KernelKit project
ASTRA KernelKit is part of the `ASTRA Toolbox` project.

> [!NOTE] 
> Last update is August 8th to verison 1.0.0rc2. This contains updated 3D conebeam
> forward and backprojectors and some documentation. Parallel beam and more examples
> are to follow soon.
> 
> Find documentation at [https://kernelkit.gra.as/](https://kernelkit.gra.as/).
> 

# ASTRA KernelKit

ASTRA KernelKit is an all-Python tomographic reconstruction package, leveraging 
the GPU-accelerated tomographic projection kernels of the [ASTRA Toolbox](https://github.com/astra-toolbox/astra-toolbox) 
using [CuPy](https://cupy.dev/).
 - Develop, explore and debug algorithms and projectors with the Python ecosystem.
 - Use CUDA features such as graphs and unified memory for maximum performance.
 - Kernels are compiled during the script with NVRTC and [Jinja2](https://jinja.palletsprojects.com/). No 
   complicated build system, and yet easy customization of kernels.
 - Easy to install and integrates well with PyTorch and other GPU frameworks.

## Installation
KernelKit depends on [CuPy](https://docs.cupy.dev/en/stable/install.html), which
we recommend to be installed first. Refer to our [installation instructions](https://kernelkit.gra.as/general/install.html).

## Getting started
See the [documentation](https://kernelkit.gra.as) for more information.

## License
The ASTRA KernelKit and ASTRA Toolbox projects are freely available and open source under the GPLv3 license.
See the [LICENSE.md](LICENSE.md) file for details.

## Contact
Please use Github pull requests (contributions welcome!) and issue tracker for bug reports and feature requests. For other
questions, please contact the authors.

Copyright (c) 2010-2023, imec [Vision Lab](http://visielab.uantwerpen.be/),
University of Antwerp 2014-2023, [CWI](https://www.cwi.nl/), Amsterdam.
