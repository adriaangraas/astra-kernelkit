import cupy as cp
import numpy as np
import pytest

from astrapy.geom3d import AstraStatic3DGeometry, Flat2DDetector
from astrapy.kernel import _copy_to_texture

