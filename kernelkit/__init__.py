__version__ = "1.0.0-alpha.2"

KERNELKIT_VERSION = __version__
KERNELKIT_CUDA_SOURCES = "kernelkit.cuda"

from .operator import *
from .algo import fp, bp, fdk, sirt
from .data import *
from .geom import resolve_volume_geometry
from .geom.proj import *
from .geom.vol import *
from . import kernel
from . import kernels
from . import experimental
from . import processing

# don't import automatically to avoid ASTRA dependency
# from . import toolbox_support
