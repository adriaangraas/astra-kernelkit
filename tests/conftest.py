import cupy as cp
import numpy as np
import pytest

from astrapy.geom.proj import ProjectionGeometry, Detector
from astrapy.kernel import copy_to_texture

