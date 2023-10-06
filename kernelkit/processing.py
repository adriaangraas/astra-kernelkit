"""
Where marked with a comment, the source is taken, and/or modified
from the scikit-image team, in particular the iradon function.
The license is a Modified BSD, which is stated below:

Copyright (C) 2011, the scikit-image team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
 3. Neither the name of skimage nor the names of its contributors may be
    used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from functools import lru_cache

import cupy as cp
import numpy as np
from tqdm import tqdm

from kernelkit.geom.proj import ProjectionGeometry


@lru_cache
def _filter(num, filter_name: str = "ramlak", use_cupy: bool = True):
    """Filter in Fourier domain
    https://gray.mgh.harvard.edu/attachments/article/166/166_HST_S14_lect1_v2.pdf

    This function is a modification of the scikit-image project.
    https://github.com/scikit-image/scikit-image/blob/main/skimage/transform/radon_transform.py

    It is better to build the Ram-Lak (ramp) filter in the spatial
    domain, rather than in the Fourier domain. This prevents to some extent
    the introduction of discretization errors. These would otherwise show
    up as a "cupping effect" in the reconstruction.

    See Kak & Slaney, Chapter 3.

    Parameters
    ----------
    num : int
        The number of samples in the filter.
    filter_name : str, optional
        The name of the filter to use. Can be one of "ramp", "ramlak",
        "cosine". If you need a different filter please file an issue.
    use_cupy : bool, optional
        Whether to use cupy or numpy.

    Returns
    -------
    four_filter : array-like
        The filter in the Fourier domain.
    """
    assert num % 2 == 0, "Filter must be even."
    xp = cp if use_cupy else np
    n1 = xp.arange(0, num / 2 + 1, dtype=xp.int32)
    n2 = xp.arange(num / 2 - 1, 0, -1, dtype=xp.int32)
    n = xp.concatenate((n1, n2))
    g = xp.zeros(len(n))
    g[0] = 0.25
    g[1::2] = -1 / (xp.pi * n[1::2]) ** 2
    four_filter = 4 * xp.real(xp.fft.fft(g))
    if filter_name.lower() == "ramp" or filter_name.lower() == "ramlak":
        pass
    elif filter_name.lower() == "cosine":
        freq = xp.linspace(0, xp.pi, num, endpoint=False)
        cosine_filter = xp.fft.fftshift(xp.sin(freq))
        four_filter *= cosine_filter
    else:
        raise ValueError(
            f"Filter with `filter_name` {filter_name} not known or not yet implemented."
        )
    return four_filter


def filter(projections, verbose: bool = False, filter: str = "ramlak"):
    """Filter in Fourier domain

    Parameters
    ----------
    projections : list of numpy.ndarray or cupy.ndarray
        The projections to filter.
    verbose : bool, optional
        Whether to show a progress bar.
    filter : str, optional
        The name of the filter to use. Can be one of "ramp", "ramlak", "cosine".
        If you need a different filter please file an issue.
    """
    xp = cp.get_array_module(projections[0])
    # this function follows structurally scikit-image's iradon()
    padding_shape = max(
        64, int(2 ** int(xp.ceil(xp.log2(2 * projections[0].shape[1]))))
    )
    four_filt = _filter(padding_shape, filter_name=filter, use_cupy=xp == cp)
    p_tmp = xp.empty((projections[0].shape[0], padding_shape))
    for p in tqdm(projections, desc="Filtering", disable=not verbose):
        assert cp.get_array_module(p) == xp, "Arrays need to be all cupy or all numpy."
        p_tmp[...] = xp.pad(
            p,
            ((0, 0), (0, padding_shape - projections[0].shape[1])),
            mode="constant",
            constant_values=0,
        )
        f = xp.fft.fft(p_tmp) * four_filt  # complex mult
        p_tmp[...] = xp.real(xp.fft.ifft(f, n=p_tmp.shape[1]))
        p[...] = p_tmp[..., : p.shape[1]]


def _ramlak_filter_fourier(projections: cp.ndarray, verbose=False):
    """Filter in Fourier domain

    I have left this function in for reference, but it is not used. It is
    known to introduce discretization errors.

    Parameters
    ----------
    projections : list of numpy.ndarray or cupy.ndarray
    """
    xp = cp.get_array_module(projections[0])
    ramlak = xp.linspace(-1, 1, num=projections[0].shape[1] // 2 + 1)
    ramlak = xp.abs(ramlak)
    for p in tqdm(projections, desc="Filtering", disable=not verbose):
        assert cp.get_array_module(p) == xp, "Arrays need to be all cupy or all numpy."
        f = xp.fft.fftshift(xp.fft.rfft(p))
        f *= ramlak  # complex mult with ramp filter
        p[...] = xp.fft.irfft(xp.fft.ifftshift(f), n=p.shape[1])


def preweight(
    projections: cp.ndarray,
    geoms: list[ProjectionGeometry],
    detector_piercings: list = None,
    verbose: bool = False,
):
    """Pixelwise rescaling to compensate for ray length in conebeam images

    Parameters
    ----------
    projections : list of numpy.ndarray or cupy.ndarray
        The projections to filter.
    geoms : list of ProjectionGeometry
        The geometries of the projections.
    detector_piercings : list of numpy.ndarray, optional
        The piercing points of the central ray through the detector. If not
        given, the detector midpoint position is used.
    verbose : bool, optional
        Whether to show a progress bar.
    """
    xp = cp.get_array_module(projections[0])
    rows, cols = xp.mgrid[0 : geoms[0].detector.rows, 0 : geoms[0].detector.cols]
    rows_view = xp.repeat(rows[:, :, xp.newaxis], 3, 2)
    cols_view = xp.repeat(cols[:, :, xp.newaxis], 3, 2)

    for i, (p, g) in enumerate(
        tqdm(zip(projections, geoms), disable=not verbose, desc="Preweighting")
    ):
        assert cp.get_array_module(p) == xp, "Arrays need to be all cupy or all numpy."
        piercing_point = (
            g.detector_position if detector_piercings is None else detector_piercings[i]
        )
        central_ray = np.linalg.norm(piercing_point - g.source_position)
        pixels = (
            xp.array(g.detector_extent_min)
            + cols_view * xp.array(g.u * g.detector.pixel_width)
            + rows_view * xp.array(g.v * g.detector.pixel_height)
        )
        pixel_rays = xp.linalg.norm(pixels - xp.array(g.source_position), axis=2)
        p *= xp.divide(central_ray, pixel_rays)
