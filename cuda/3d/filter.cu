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
extern "C" {
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
        const int startRow = (
                blockIdx.x / ((nrCols  + {{ det_block_u }} - 1) / {{ det_block_u }})
            ) * {{ det_block_v }};

        int endRow = startRow + {{ det_block_v }};
        if (endRow > nrRows)
            endRow = nrRows;

        // We need the length of the central ray and the length of the ray(s) to
        // our detector pixel(s).
        const float U = (column - .5f * nrCols  + .5f) * detUSize;
        float V = (startRow - .5f * nrRows + .5f) * detVSize + zShift;
        const float centralRayLength = srcOrigin + detOrigin;
        const float T = centralRayLength * centralRayLength + U * U;

        // Contributions to the weighting factors:
        // fCentralRayLength / fRayLength   : the main FDK preweighting factor
        // fSrcOrigin / (fDetUSize * fCentralRayLength)
        //                                  : to adjust the filter to the det width
        // pi / (2 * iProjAngles)           : scaling of the integral over angles
        const float W2 = centralRayLength / (detUSize * srcOrigin);
        const float W = centralRayLength * W2 * (M_PI / 2.f) / (float) nrProjections;

        for (int row = startRow; row < endRow; ++row) {
            const float rayLength = sqrtf(T + V * V);
            projections[angle][row * nrCols + column] *= W / rayLength;
            V += detVSize;
        }
    }
}