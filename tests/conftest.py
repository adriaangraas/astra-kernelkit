import cupy as cp
import numpy as np
import pytest

from kernelkit.geom.proj import ProjectionGeometry, Detector
from kernelkit.kernel import copy_to_texture

