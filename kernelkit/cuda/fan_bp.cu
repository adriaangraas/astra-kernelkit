/*
-----------------------------------------------------------------------
Copyright: 2010-2018, imec Vision Lab, University of Antwerp
           2014-2018, CWI, Amsterdam

Contact: astra@astra-toolbox.com
Website: http://www.astra-toolbox.com/

This file is part of the ASTRA Toolbox.


The ASTRA Toolbox is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The ASTRA Toolbox is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.

-----------------------------------------------------------------------
*/
struct Params { float numC, numX, numY, denC, denX, denY; };

__constant__ Params params[{{ nr_projs_global }}];

/**
  * @brief Backprojection kernel for the fan-beam geometry.
  *
  * @param projTexture The texture containing the projections.
  * @param start The index of the first projection to backproject.
  * @param nrProjections The total number of projections to backproject from.
  * @param volume The volume to backproject into.
  * @param volWidth The width of the volume.
  * @param volHeight The height of the volume.
  * @param outputScale The scale factor to apply to the output.
  *
  * @note The kernel contains three Jinja2 template parameters:
  *       - nr_projs_block: The number of projections to backproject per block.
  *       - nr_vxls_block_x: The number of voxels to backproject per block in the x-direction.
  *       - nr_vxls_block_y: The number of voxels to backproject per block in the y-direction.
  */
__global__ void fan_bp(
    float * volume,
    cudaTextureObject_t projTexture,
    int start,
    int nrProjections,
    int voxelsX,
    int voxelsY,
    int dimY, // != voxelsY, as potentially pitched dimension
    float outputScale
) {
	int end = start + {{ nr_projs_block }};
	if (end > nrProjections)
		end = nrProjections;

    // x, y are the coordinates of the voxel in the volume.
	const int X = blockIdx.x * {{ nr_vxls_block_x }} + threadIdx.x;
	const int Y = blockIdx.y * {{ nr_vxls_block_y }} + threadIdx.y;

    // Threads outside the volume do not need to do anything.
	if (X >= voxelsX || Y >= voxelsY)
		return;

	// Coordinates of projection start in the center of the voxel.
	const float x = (X - .5 * voxelsX + .5);
	const float y = -(Y - .5 * voxelsY + .5);

    // Accumulate all pixel interpolations in `v`.
    float v = .0;
	float a = start + .5; // index for texture must be float + .5

	for (int i = start; i < end; ++i) {
		// The scale factor is the approximate number of rays traversing this
		// pixel, given by the inverse size of a detector pixel scaled by
		// the magnification factor of this pixel. Magnification factor is
		// || u (d-s) || / || u (x-s) ||
		const Params & p = params[i];
		float r = __fdividef(1., p.denC + p.denX * x + p.denY * y);
//		                         (FBPWEIGHT ? 1.0 : p.denC) + p.denX * x + p.denY * y);
		float T = (p.numC + p.numX * x + p.numY * y) * r;
		v += tex2D<float>(projTexture, T, a) * r;
//		                                     * (FBPWEIGHT ? r * r : r);
		a += 1.; // update index for texture
	}

	// Write the result to the volume. Note that `x` is the fastest varying
	// dimension in the volume, so we can use coalesced writes.
//	volume[Y * dimY + X] += v * outputScale;
	volume[Y * dimY + X] += 1.;
}