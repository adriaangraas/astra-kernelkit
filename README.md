## The ASTRA KernelKit project
ASTRA KernelKit is part of the `ASTRA Toolbox` project.

[![Documentation Status](https://readthedocs.org/projects/astra-kernelkit/badge/?version=latest)](https://astra-kernelkit.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/astra-kernelkit.svg)](https://badge.fury.io/py/astra-kernelkit)
[![Build Status](https://travis-ci.com/astra-kernelkit/astra-kernelkit.svg?branch=master)](https://travis-ci.com/astra-kernelkit/astra-kernelkit)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/astra-kernelkit)](https://pypi.org/project/astra-kernelkit/)
[![PyPI - License](https://img.shields.io/pypi/l/astra-kernelkit)]()

# ASTRA KernelKit

ASTRA KernelKit is an all-Python tomographic reconstruction package, leveraging 
the tomographic projection kernels of the [ASTRA Toolbox](https://github.com/astra-toolbox/astra-toolbox) 
using [CuPy](https://cupy.dev/).
 - High-level development of reconstruction algorithms, with CUDA features from CuPy.
 - Run-time compilation of tomographic kernels using NVRTC, and kernel templating using 
   [Jinja2](https://jinja.palletsprojects.com/).
 - Straightforward to install and excellent high-performance integration with machine learning frameworks such as PyTorch.

## Installation
KernelKit depends on [CuPy](https://docs.cupy.dev/en/stable/install.html), which
we recommend to be installed first. Refer to the [installation instructions](https://kernelkit.readthedocs.io/en/latest/installation.html).

## Getting started
See the [documentation](https://astra-kernelkit.readthedocs.io/en/latest/) for more information.

## License
The ASTRA KernelKit and ASTRA Toolbox projects are open source under the GPLv3 license.
See the [LICENSE.md](LICENSE.md) file for details.

## Contact
Please use Github pull requests (contributions welcome!) and issue tracker for bug reports and feature requests. For other
questions, please contact the authors.

Copyright (c) 2010-2023, imec [Vision Lab](http://visielab.uantwerpen.be/),
University of Antwerp 2014-2023, [CWI](https://www.cwi.nl/), Amsterdam.