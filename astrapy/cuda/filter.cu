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
#define M_PI           3.14159265358979323846  /* pi */

extern "C" {
/**
 * TODO(ASTRA): To support non-cube voxels, preweighting needs per-view
 * parameters. NB: Need to properly take into account the
 * anisotropic volume normalization done for that too.
 */
__global__
void preweight(
    float ** projections,
    unsigned int start,
    unsigned int end,
    float srcPosition,
    float detPosition,
    float zShift,
    float pixelWidth,
    float pixelHeight,
    int nrProjections,
    int nrCols,
    int nrRows
) {
    int a = start
        + blockIdx.y * {{angles_per_weight_block}}
        + threadIdx.y;

    if (a >= end)
        return;

    const int column = (
            blockIdx.x % ((nrCols  + {{ det_block_u }} - 1) / {{ det_block_u }})
        ) * {{ det_block_v }} + threadIdx.x;
    const int startRow = (
            blockIdx.x / ((nrCols  + {{ det_block_u }} - 1) / {{ det_block_u }})
        ) * {{ det_block_v }};

    int endRow = startRow + {{ det_block_v }};
    if (endRow > nrRows)
        endRow = nrRows;

    // We need the length of the central ray and the length of the ray(s) to
    // our detector pixel(s).
    const float U = (column - .5f * nrCols  + .5f) * pixelWidth;
    float V = (startRow - .5f * nrRows + .5f) * pixelHeight + zShift;
    const float centralRayLength = srcPosition + detPosition;
    const float T = centralRayLength * centralRayLength + U * U;

    // Contributions to the weighting factors:
    // fCentralRayLength / fRayLength
    //      the main FDK preweighting factor
    // fSrcOrigin / (fDetUSize * fCentralRayLength)
    //      to adjust the filter to the det width
    // pi / (2 * iProjAngles)
    //      scaling of the integral over angles
    const float W2 = centralRayLength / (pixelWidth * srcPosition);
    const float W = centralRayLength * W2 * (M_PI / 2.f) / (float) nrProjections;
    for (int row = startRow; row < endRow; ++row) {
        const float rayLength = sqrtf(T + V * V);
        projections[a][row * nrCols + column] *= W / rayLength;
        V += pixelHeight;
    }
}
}