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
typedef texture<float, 3, cudaReadModeElementType> texture3D;

__constant__ float srcsX[{{ max_angles }}];
__constant__ float srcsY[{{ max_angles }}];
__constant__ float srcsZ[{{ max_angles }}];
__constant__ float detsSX[{{ max_angles }}];
__constant__ float detsSY[{{ max_angles }}];
__constant__ float detsSZ[{{ max_angles }}];
__constant__ float detsUX[{{ max_angles }}];
__constant__ float detsUY[{{ max_angles }}];
__constant__ float detsUZ[{{ max_angles }}];
__constant__ float detsVX[{{ max_angles }}];
__constant__ float detsVY[{{ max_angles }}];
__constant__ float detsVZ[{{ max_angles }}];

// x=0, y=1, z=2
struct DIR_X {
    __device__ float nSlices(unsigned int volX,
                             unsigned int volY,
                             unsigned int volZ) const { return volX; }
    __device__ float nDim1(unsigned int volX,
                           unsigned int volY,
                           unsigned int volZ) const { return volY; }
    __device__ float nDim2(unsigned int volX,
                           unsigned int volY,
                           unsigned int volZ) const { return volZ; }
    __device__ float x(float x, float y, float z) const { return x; }
    __device__ float y(float x, float y, float z) const { return y; }
    __device__ float z(float x, float y, float z) const { return z; }
    __device__ float tex(texture3D vol, float f0, float f1, float f2) const {
        return tex3D(vol, f0, f1, f2); }
};
// y=0, x=1, z=2
struct DIR_Y {
    __device__ float nSlices(unsigned int volX,
                             unsigned int volY,
                             unsigned int volZ) const { return volY; }
    __device__ float nDim1(unsigned int volX,
                           unsigned int volY,
                           unsigned int volZ) const { return volX; }
    __device__ float nDim2(unsigned int volX,
                           unsigned int volY,
                           unsigned int volZ) const { return volZ; }
    __device__ float x(float x, float y, float z) const { return y; }
    __device__ float y(float x, float y, float z) const { return x; }
    __device__ float z(float x, float y, float z) const { return z; }
    __device__ float tex(texture3D vol, float f0, float f1, float f2) const {
        return tex3D(vol, f1, f0, f2); }
};

// z=0, x=1, y=2
struct DIR_Z {
    __device__ float nSlices(unsigned int volX,
                             unsigned int volY,
                             unsigned int volZ) const { return volZ; }
    __device__ float nDim1(unsigned int volX,
                           unsigned int volY,
                           unsigned int volZ) const { return volX; }
    __device__ float nDim2(unsigned int volX,
                           unsigned int volY,
                           unsigned int volZ) const { return volY; }
    __device__ float x(float x, float y, float z) const { return z; }
    __device__ float y(float x, float y, float z) const { return x; }
    __device__ float z(float x, float y, float z) const { return y; }
    __device__ float tex(texture3D vol, float f0, float f1, float f2) const {
        return tex3D(vol, f1, f2, f0); }
};


/**
 * Forward projection for cone geometry with flat detector
 * TODO(Adriaan):
 *  - scaling distCorr can be slightly faster since scale1=1 and
 *    scale2=1 for volumes that scale as a cube
 *  - bring supersampling back
 */
template<typename COORD>
__global__ void cone_fp(
    texture3D volumeTexture,
    float * projections,
    unsigned int startSlice,
    unsigned int startAngle,
    unsigned int endAngle,
    unsigned int volX,
    unsigned int volY,
    unsigned int volZ,
    unsigned int projAngles,
    unsigned int detectorCols,
    unsigned int detectorRows,
    float scale1,
    float scale2,
    float outputScale
) {
    COORD c;

    int angle = startAngle + blockIdx.y * {{ angles_per_block }} + threadIdx.y;
    if (angle >= endAngle)
        return;

    const float sX = srcsX[angle];
    const float sY = srcsY[angle];
    const float sZ = srcsZ[angle];
    const float dUX = detsUX[angle];
    const float dUY = detsUY[angle];
    const float dUZ = detsUZ[angle];
    const float dVX = detsVX[angle];
    const float dVY = detsVY[angle];
    const float dVZ = detsVZ[angle];
    const float dSX = detsSX[angle] + .5f * dUX + .5f * dVX;
    const float dSY = detsSY[angle] + .5f * dUY + .5f * dVY;
    const float dSZ = detsSZ[angle] + .5f * dUZ + .5f * dVZ;

    const int rowsInBlock = (detectorCols + {{ columns_per_block }} - 1)
                            / {{ columns_per_block }};
    const int column = (blockIdx.x % rowsInBlock) * {{ columns_per_block }}
                         + threadIdx.x;
    const int startRow = (blockIdx.x / rowsInBlock) * {{ rows_per_block }};

    int endRow = startRow + {{ rows_per_block }};
    if (endRow > detectorRows)
        endRow = detectorRows;

    int endSlice = startSlice + {{ block_slices }};
    if (endSlice > c.nSlices(volX, volY, volZ))
        endSlice = c.nSlices(volX, volY, volZ);

    // Trace the ray through a slice of the volume.
    for (int row = startRow; row < endRow; ++row) {
        const float detX = dSX + column * dUX + row * dVX;
        const float detY = dSY + column * dUY + row * dVY;
        const float detZ = dSZ + column * dUZ + row * dVZ;

        /*        (x)   ( 1)       ( 0) *
         * ray:   (y) = (ay) * x + (by) *
         *        (z)   (az)       (bz) */
        const float a1 = (c.y(sX, sY, sZ) - c.y(detX, detY, detZ)) /
                         (c.x(sX, sY, sZ) - c.x(detX, detY, detZ));
        const float a2 = (c.z(sX, sY, sZ) - c.z(detX, detY, detZ)) /
                         (c.x(sX, sY, sZ) - c.x(detX, detY, detZ));
        const float b1 = c.y(sX, sY, sZ) - a1 * c.x(sX, sY, sZ);
        const float b2 = c.z(sX, sY, sZ) - a2 * c.x(sX, sY, sZ);

        /* Could be (for cubes):
         *  sqrt(a1 * a1 + a2 * a2 + 1.0f) * outputScale; */
        const float distCorrection = sqrt(a1 * a1 * scale1 + a2 * a2 * scale2 + 1.f)
             * outputScale;

        // coord is a coordinate in real space
        float coord0 = startSlice + .5f;
        float coord1 = a1 * (startSlice - .5f * c.nSlices(volX, volY, volZ) + .5f)
            + b1 + .5f * c.nDim1(volX, volY, volZ);
        float coord2 = a2 * (startSlice - .5f * c.nSlices(volX, volY, volZ) + .5f)
            + b2 + .5f * c.nDim2(volX, volY, volZ);


        float val = 0.f;
        for (int s = startSlice; s < endSlice; ++s) {
            // add interpolated voxel value at current coordinate
            val += c.tex(volumeTexture, coord0, coord1, coord2);
            coord0 += 1.f;
            coord1 += a1;
            coord2 += a2;
//            if (startRow == 0 and column == 0 and angle == 0 and s == startSlice) {
//            printf("%f %f %f %f\n", coord0, coord1, coord2, val);
//            }
        }
        val *= distCorrection;

        projections[angle * detectorRows * detectorCols
                    + column * detectorRows
                    + row] += val;
    }
}