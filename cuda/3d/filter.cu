#define M_PI           3.14159265358979323846  /* pi */

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
//__global__
//static void applyFilter(
//    int _iProjectionCount,
//    int _iFreqBinCount,
//    cufftComplex * sinogram,
//    cufftComplex * filter
//) {
//	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
//	const int iProjectionIndex = idx / _iFreqBinCount;
//
//	if (iProjectionIndex >= _iProjectionCount)
//		return;
//
//	float A = sinogram[idx].x;
//	float B = sinogram[idx].y;
//	float C = filter[idx].x;
//	float D = filter[idx].y;
//
//    // I think this is just complex multiplications
//	sinogram[idx].x = A * C - B * D;
//	sinogram[idx].y = A * D + C * B;
//}

extern "C" {

__global__
void rescaleInverseFourier(
    int _iProjectionCount,
    int _iDetectorCount,
    float * _pfInFourierOutput
) {
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int iProjectionIndex = idx / _iDetectorCount;
	const int iDetectorIndex = idx % _iDetectorCount;

	if (iProjectionIndex >= _iProjectionCount)
		return;

	_pfInFourierOutput[iProjectionIndex * _iDetectorCount + iDetectorIndex]
        /= (float) _iDetectorCount;
}

/**
 * TODO(ASTRA): To support non-cube voxels, preweighting needs per-view
 * parameters. NB: Need to properly take into account the
 * anisotropic volume normalization done for that too.
 */
__global__
void preweight(
    float ** projections,
    unsigned int startAngle,
    unsigned int endAngle,
    float srcOrigin,
    float detOrigin,
    float zShift,
    float detUSize,
    float detVSize,
    int nrProjections,
    int nrCols,
    int nrRows
) {
	int angle = startAngle
        + blockIdx.y * {{angles_per_weight_block}}
        + threadIdx.y;

	if (angle >= endAngle)
		return;

	const int column = (
            blockIdx.x % ((nrCols  + {{ det_block_u }} - 1) / {{ det_block_u }})
        ) * {{ det_block_v }} + threadIdx.x;
	const int startDetectorV = (
            blockIdx.x / ((nrCols  + {{ det_block_u }} - 1) / {{ det_block_u }})
        ) * {{ det_block_v }};

	int endDetectorV = startDetectorV + {{ det_block_v }};
	if (endDetectorV > nrRows)
		endDetectorV = nrRows;

	// We need the length of the central ray and the length of the ray(s) to
	// our detector pixel(s).
	const float U = (column - .5f * nrCols  + .5f) * detUSize;
	float V = (startDetectorV - .5f * nrRows + .5f) * detVSize + zShift;
	const float centralRayLength = srcOrigin + detOrigin;
	const float T = centralRayLength * centralRayLength + U * U;

	// Contributions to the weighting factors:
	// fCentralRayLength / fRayLength   : the main FDK preweighting factor
	// fSrcOrigin / (fDetUSize * fCentralRayLength)
	//                                  : to adjust the filter to the det width
	// pi / (2 * iProjAngles)           : scaling of the integral over angles
	const float W2 = centralRayLength / (detUSize * srcOrigin);
	const float W = centralRayLength * W2 * (M_PI / 2.f) / (float) nrProjections;

	for (int row = startDetectorV; row < endDetectorV; ++row) {
		const float rayLength = sqrtf(T + V * V);
		const float weight = W / rayLength;
//		projData[(row * nrProjections + angle) * projPitch
//		         + column] *= weight;
        projections[angle][row * nrCols + column] *= weight;
		V += detVSize;
	}
}

}