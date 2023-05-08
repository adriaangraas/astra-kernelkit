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
    __device__ inline static float nSlices(unsigned int voxX, unsigned int voxY, unsigned int voxZ) {
         return voxX; }
    __device__ inline static float nDim1(unsigned int volX, unsigned int voxY, unsigned int voxZ) {
        return voxY; }
    __device__ inline static float nDim2(unsigned int voxX, unsigned int voxY, unsigned int voxZ) {
        return voxZ; }
    __device__ inline static float x(float x, float y, float z) { return x; }
    __device__ inline static float y(float x, float y, float z) { return y; }
    __device__ inline static float z(float x, float y, float z) { return z; }
    __device__ inline static float tex(const cudaTextureObject_t& vol, float f0, float f1, float f2) {
        return tex3D<float>(vol, f0, f1, f2); }
};

// y=0, x=1, z=2
struct DIR_Y {
    __device__ inline static float nSlices(unsigned int volX, unsigned int voxY, unsigned int voxZ) {
        return voxY; }
    __device__ inline static float nDim1(unsigned int voxX, unsigned int voxY, unsigned int voxZ) {
        return voxX; }
    __device__ inline static float nDim2(unsigned int voxX, unsigned int voxY, unsigned int voxZ) {
        return voxZ; }
    __device__ inline static float x(float x, float y, float z) { return y; }
    __device__ inline static float y(float x, float y, float z) { return x; }
    __device__ inline static float z(float x, float y, float z) { return z; }
    __device__ inline static float tex(const cudaTextureObject_t& vol, float f0, float f1, float f2) {
        return tex3D<float>(vol, f0, f2, f1); } //  1 0 2
};

struct DIR_Z { // z=0, x=1, y=2
    __device__ inline static float nSlices(unsigned int volX, unsigned int voxY, unsigned int voxZ) {
        return voxZ; }
    __device__ inline static float nDim1(unsigned int voxX, unsigned int voxY, unsigned int voxZ) {
        return voxX; }
    __device__ inline static float nDim2(unsigned int voxX, unsigned int voxY, unsigned int voxZ) {
        return voxY; }
    __device__ inline static float x(float x, float y, float z) { return z; }
    __device__ inline static float y(float x, float y, float z) { return x; }
    __device__ inline static float z(float x, float y, float z) { return y; }
    __device__ inline static float tex(const cudaTextureObject_t& vol, float f0, float f1, float f2) {
        return tex3D<float>(vol, f2, f0, f1); }  // 1 2 0
};

/**
 * Forward projection for cone geometry with flat detector
 * TODO(Adriaan):
 *  - scaling distCorr can be slightly faster since scale1=1 and
 *    scale2=1 for volumes that scale as a cube
 *  - bring supersampling back
 *
 * A thread corresponds to a (row,) and computes a column portion for a geometry.
 * A block is (number of rows, number of column portions, number of projections)
 */
template<typename C>
__global__ void cone_fp(
    cudaTextureObject_t volumeTexture,
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
    int row;
    int col;
    int endRow;
    int endCol;
    const int pixelsPerThread = {{ pixels_per_thread }};

    {% if mode_row %}
        row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= rows)
            return;
        endRow = row + 1;

        col = blockIdx.y * pixelsPerThread;
        endCol = col + pixelsPerThread;
        if (endCol > cols)
            endCol = cols;
    {% else %}
        col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= cols)
            return;
        endCol = col + 1;

        row = blockIdx.y * pixelsPerThread;
        endRow = row + pixelsPerThread;
        if (endRow > rows)
            endRow = rows;
    {% endif %}

    const int proj = offsetProj + blockIdx.z;

    int endSlice = offsetSlice + {{ slices_per_thread }};
    if (endSlice > C::nSlices(voxX, voxY, voxZ))
        endSlice = C::nSlices(voxX, voxY, voxZ);

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

    for (int r = row; r < endRow; ++r) {
//        printf("row %d \n ", row);
        for (int c = col; c < endCol; ++c) {
            // Trace the ray through a slice of the volume.
            const float detX = dSX + c * dUX + r * dVX;
            const float detY = dSY + c * dUY + r * dVY;
            const float detZ = dSZ + c * dUZ + r * dVZ;

            /*        (x)   ( 1)       ( 0) *
             * ray:   (y) = (ay) * x + (by) *
             *        (z)   (az)       (bz) */
            const float a1 = (C::y(sX, sY, sZ) - C::y(detX, detY, detZ)) / (C::x(sX, sY, sZ) - C::x(detX, detY, detZ));
            const float a2 = (C::z(sX, sY, sZ) - C::z(detX, detY, detZ)) / (C::x(sX, sY, sZ) - C::x(detX, detY, detZ));
            const float b1 = C::y(sX, sY, sZ) - a1 * C::x(sX, sY, sZ);
            const float b2 = C::z(sX, sY, sZ) - a2 * C::x(sX, sY, sZ);
            const int nSlices = C::nSlices(voxX, voxY, voxZ);
            const int nDim1 = C::nDim1(voxX, voxY, voxZ);
            const int nDim2 = C::nDim2(voxX, voxY, voxZ);
            float x = offsetSlice + .5f; // ray direction
            float y = a1 * (offsetSlice - .5f * nSlices + .5f) + b1 + .5f * nDim1;
            float z = a2 * (offsetSlice - .5f * nSlices + .5f) + b2 + .5f * nDim2;

            float val = 0.f;
            for (int s = offsetSlice; s < endSlice; ++s) {
                // add interpolated voxel value at current coordinate
                val += C::tex(volumeTexture, z, y, x);
                x += 1.f;
                y += a1;
                z += a2;
            }

            /* Distance correction: Could be (for cubes):
             *  sqrt(a1 * a1 + a2 * a2 + 1.0f) * outputScale; */
            val *= sqrt(a1 * a1 * scale1 + a2 * a2 * scale2 + 1.f);
            // printf("row %d col %d rows %d cols %d\n %d\n ", row, col, rows, cols, val);
            {% if projs_row_major %}
                // faster if kernel is in MODE_ROW
                projections[proj][r * cols + c] += val * outputScale;
            {% else %}
                // faster if kernel is not in MODE_ROW
                projections[proj][c * rows + r] += val * outputScale;
            {% endif %}
        }
    }
}