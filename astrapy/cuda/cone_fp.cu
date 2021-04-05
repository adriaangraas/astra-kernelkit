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

__constant__ float srcsX[{{ nr_projs_global }}];
__constant__ float srcsY[{{ nr_projs_global }}];
__constant__ float srcsZ[{{ nr_projs_global }}];
__constant__ float detsSX[{{ nr_projs_global }}];
__constant__ float detsSY[{{ nr_projs_global }}];
__constant__ float detsSZ[{{ nr_projs_global }}];
__constant__ float detsUX[{{ nr_projs_global }}];
__constant__ float detsUY[{{ nr_projs_global }}];
__constant__ float detsUZ[{{ nr_projs_global }}];
__constant__ float detsVX[{{ nr_projs_global }}];
__constant__ float detsVY[{{ nr_projs_global }}];
__constant__ float detsVZ[{{ nr_projs_global }}];

// x=0, y=1, z=2
struct DIR_X {
    __device__ float nSlices(unsigned int voxX,
                             unsigned int voxY,
                             unsigned int voxZ) const { return voxX; }
    __device__ float nDim1(unsigned int volX,
                           unsigned int voxY,
                           unsigned int voxZ) const { return voxY; }
    __device__ float nDim2(unsigned int voxX,
                           unsigned int voxY,
                           unsigned int voxZ) const { return voxZ; }
    __device__ float x(float x, float y, float z) const { return x; }
    __device__ float y(float x, float y, float z) const { return y; }
    __device__ float z(float x, float y, float z) const { return z; }
    __device__ float tex(texture3D vol, float f0, float f1, float f2) const {
        return tex3D(vol, f0, f1, f2); }
};
// y=0, x=1, z=2
struct DIR_Y {
    __device__ float nSlices(unsigned int volX,
                             unsigned int voxY,
                             unsigned int voxZ) const { return voxY; }
    __device__ float nDim1(unsigned int voxX,
                           unsigned int voxY,
                           unsigned int voxZ) const { return voxX; }
    __device__ float nDim2(unsigned int voxX,
                           unsigned int voxY,
                           unsigned int voxZ) const { return voxZ; }
    __device__ float x(float x, float y, float z) const { return y; }
    __device__ float y(float x, float y, float z) const { return x; }
    __device__ float z(float x, float y, float z) const { return z; }
    __device__ float tex(texture3D vol, float f0, float f1, float f2) const {
        return tex3D(vol, f0, f2, f1); } //  1 0 2
};

// z=0, x=1, y=2
struct DIR_Z {
    __device__ float nSlices(unsigned int volX,
                             unsigned int voxY,
                             unsigned int voxZ) const { return voxZ; }
    __device__ float nDim1(unsigned int voxX,
                           unsigned int voxY,
                           unsigned int voxZ) const { return voxX; }
    __device__ float nDim2(unsigned int voxX,
                           unsigned int voxY,
                           unsigned int voxZ) const { return voxY; }
    __device__ float x(float x, float y, float z) const { return z; }
    __device__ float y(float x, float y, float z) const { return x; }
    __device__ float z(float x, float y, float z) const { return y; }
    __device__ float tex(texture3D vol, float f0, float f1, float f2) const {
        return tex3D(vol, f2, f0, f1); }  // 1 2 0
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
    float ** projections,
    unsigned int offsetSlice,
    unsigned int offsetProj,
    unsigned int voxX,
    unsigned int voxY,
    unsigned int voxZ,
    unsigned int rows,
    unsigned int cols,
    float scale1,
    float scale2,
    float outputScale
) {
    COORD c;

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows)
        return;

    int col = blockIdx.y * {{ cols_per_thread }};
    int endCol = col + {{ cols_per_thread }};
    if (endCol > cols)
        endCol = cols;

    const int proj = offsetProj + blockIdx.z;

    int endSlice = offsetSlice + {{ slices_per_thread }};
    if (endSlice > c.nSlices(voxX, voxY, voxZ))
        endSlice = c.nSlices(voxX, voxY, voxZ);

    const float sX = srcsX[proj];
    const float sY = srcsY[proj];
    const float sZ = srcsZ[proj];
    const float dUX = detsUX[proj];
    const float dUY = detsUY[proj];
    const float dUZ = detsUZ[proj];
    const float dVX = detsVX[proj];
    const float dVY = detsVY[proj];
    const float dVZ = detsVZ[proj];
    const float dSX = detsSX[proj] + .5f * dUX + .5f * dVX;
    const float dSY = detsSY[proj] + .5f * dUY + .5f * dVY;
    const float dSZ = detsSZ[proj] + .5f * dUZ + .5f * dVZ;

    // Trace the ray through a slice of the volume.
    for (; col < endCol; ++col) {
        const float detX = dSX + col * dUX + row * dVX;
        const float detY = dSY + col * dUY + row * dVY;
        const float detZ = dSZ + col * dUZ + row * dVZ;

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

        const int nSlices = c.nSlices(voxX, voxY, voxZ);
        const int nDim1 = c.nDim1(voxX, voxY, voxZ);
        const int nDim2 = c.nDim2(voxX, voxY, voxZ);

        float x = offsetSlice + .5f; // ray direction
        float y = a1 * (offsetSlice - .5f * nSlices + .5f) + b1 + .5f * nDim1;
        float z = a2 * (offsetSlice - .5f * nSlices + .5f) + b2 + .5f * nDim2;

        float val = 0.f;
        for (int s = offsetSlice; s < endSlice; ++s) {
            // add interpolated voxel value at current coordinate
            val += c.tex(volumeTexture, z, y, x);
            x += 1.f;
            y += a1;
            z += a2;
        }
        val *= distCorrection;

        projections[proj][row * cols + col] += val;
    }
}