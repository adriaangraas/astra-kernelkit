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

template<int AXIS_X, int AXIS_Y, int AXIS_Z>
struct ProjectionDirection {
    __device__ inline static int nSlices(unsigned int voxelX, unsigned int voxelY, unsigned int voxelZ) {
        return getAxis(voxelX, voxelY, voxelZ, AXIS_X);
    }
    __device__ inline static int nDim1(unsigned int voxelX, unsigned int voxelY, unsigned int voxelZ) {
        return getAxis(voxelX, voxelY, voxelZ, AXIS_Y);
    }
    __device__ inline static int nDim2(unsigned int voxelX, unsigned int voxelY, unsigned int voxelZ) {
        return getAxis(voxelX, voxelY, voxelZ, AXIS_Z);
    }
    __device__ inline static float x(float x, float y, float z) { return getAxis(x, y, z, AXIS_X); }
    __device__ inline static float y(float x, float y, float z) { return getAxis(x, y, z, AXIS_Y); }
    __device__ inline static float z(float x, float y, float z) { return getAxis(x, y, z, AXIS_Z); }
private:
    __device__ inline static float getAxis(float x, float y, float z, int axis) {
        return (axis == 0) ? x : ((axis == 1) ? y : z);
    }
};

using DIR_X = ProjectionDirection<0, 1, 2>;
using DIR_Y = ProjectionDirection<1, 0, 2>;
using DIR_Z = ProjectionDirection<2, 1, 0>;

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
    const cudaTextureObject_t volume,
    float ** __restrict__ projections,
    int offsetSlice,
    int offsetProj,
    int voxX,
    int voxY,
    int voxZ,
    int projs,
    int rows,
    int cols,
    int projsPitch,
    int rowsPitch,
    int colsPitch,
    float scale1,
    float scale2,
    float outputScale
) {
    int row, col, endRow, endCol;

    {% if projection_axes[2] == 2 %}
        // assignment: (theta->z, rows->y, cols->x)
        col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col >= cols)
            return;
        endCol = col + 1;

        row = blockIdx.y * {{ rows_per_block }};
        endRow = row + {{ rows_per_block }};
        if (endRow > rows)
            endRow = rows;

        const int proj = offsetProj + blockIdx.z;
    {% elif projection_axes[2] == 1 %}
        // assignment: (theta->z, rows->x, cols->y)
        row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= rows)
            return;
        endRow = row + 1;

        col = blockIdx.y * {{ cols_per_block }};
        endCol = col + {{ cols_per_block }};
        if (endCol >= cols)
            endCol = cols;

        const int proj = offsetProj + blockIdx.z;
    {% else %}
        // assignment: (theta->x, u->y, v->z)
        const int proj = offsetProj + blockIdx.x * blockDim.x + threadIdx.x;
        if (proj >= projs)
            return;

        row = blockIdx.y * {{ rows_per_block }};
        endRow = row + {{ rows_per_block }};
        if (endRow >= rows)
            endRow = rows;

        col = blockIdx.z * {{ cols_per_block }};
        endCol = col + {{ cols_per_block }};
        if (endCol > cols)
            endCol = cols;
    {% endif %}

    int endSlice = offsetSlice + {{ slices_per_block }};
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
        for (int c = col; c < endCol; ++c) {
            // Trace the ray through a slice of the volume.
            const float detX = dSX + c * dUX + r * dVX;
            const float detY = dSY + c * dUY + r * dVY;
            const float detZ = dSZ + c * dUZ + r * dVZ;

            /*        (x)   ( 1)       ( 0) *
             * ray:   (y) = (ay) * x + (by) *
             *        (z)   (az)       (bz) */
            // ray increments
            const float SDD = C::x(sX, sY, sZ) - C::x(detX, detY, detZ);
            const float a1 = (C::y(sX, sY, sZ) - C::y(detX, detY, detZ)) / SDD;
            const float a2 = (C::z(sX, sY, sZ) - C::z(detX, detY, detZ)) / SDD;
            // ray offsets
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
                float Cx = C::x(x, y, z);
                float Cy = C::y(x, y, z);
                float Cz = C::z(x, y, z);
                {% set ax = ['Cx', 'Cy', 'Cz'] %}
                float x_ = {{ax[volume_axes[0]]}};
                float y_ = {{ax[volume_axes[1]]}};
                float z_ = {{ax[volume_axes[2]]}};
                val += tex3D<float>(volume, z_, y_, x_);
                x += 1.f;
                y += a1;
                z += a2;
            }

            /* Distance correction: Could be (for cubes):
             *  sqrt(a1 * a1 + a2 * a2 + 1.0f) * outputScale; */
            val *= sqrt(a1 * a1 * scale1 + a2 * a2 * scale2 + 1.f);

            {% set nr = ['projsPitch', 'rowsPitch', 'colsPitch'] %}
            {% set idx = ['proj', 'r', 'c'] %}
            projections[{{idx[projection_axes[0]]}}] // projection/sinogram number
                       [{{idx[projection_axes[1]]}} * {{nr[projection_axes[2]]}} // row idx * nr of cols
                      + {{idx[projection_axes[2]]}}] // pixel in the row or col
                += val * outputScale;
        }
    }
}