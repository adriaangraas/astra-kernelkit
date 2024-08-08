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
struct Params {
    float4 numeratorU;
    float4 numeratorV;
    float4 denominator;
};

__constant__ Params params[{{ nr_projs_global }}];

__global__ void cone_bp(
{% if texture == 'tex3D' or texture == 'tex2DLayered' %}
    cudaTextureObject_t projTexture,
{% elif texture == 'surf3D' or texture == 'surf2DLayered' %}
    cudaSurfaceObject_t projSurface,
{% elif False %}
    const float ** __restrict__ projections,
    int rowsPitch,
    int colsPitch,
{% else %}
    const cudaTextureObject_t * __restrict__ projTextures,
{% endif %}
    float * __restrict__ volume,
    int start,
    int nrProjections,
    int voxels_c0,
    int voxels_c1,
    int voxels_c2,
    float outputScale
) {
    int end = start + {{ nr_projs_block }};
    if (end > nrProjections)
        end = nrProjections;

    /**
     * We have four systems:
     *  - CUDA block sizes and thread indices  (threadIdx.{x, y, z})
     *  - Voxel coordinates (c0, c1, c2) of the CuPy volume
     *  - Real-world coordinates (X, Y, Z)
     *  - Detector coordinates (u, v)
     *
     *  - The input (voxels_c0, voxels_c1, voxels_c2) directly describe the
     * sizes in contiguous memory.
     *  - Jinja2 params voxels_per_block[i] are a re-mapping of block sizes to
     *    volume dimensions, see the Python kernel. They do not
     *    correspond to volume axes.
     */

    // Retrieve current voxel coordinate from the thread indices.
    // Note `c0` is not managed by a thread index. Every thread just does some
    // extra work in the first dimension.
    const int n = (voxels_c2 + {{ voxels_per_block[2] }} - 1) / {{ voxels_per_block[2] }};
    const int start_c0 = blockIdx.y * {{ voxels_per_block[0] }};
    const int c1 = blockIdx.x / n * {{ voxels_per_block[1] }} + threadIdx.y;
    const int c2 = blockIdx.x % n * {{ voxels_per_block[2] }} + threadIdx.x;
    if (c2 >= voxels_c2 || c1 >= voxels_c1)
        return;

    // Shift voxel coordinates to center, add .5 for texture coord
    const float f_c0 = start_c0 - .5f * voxels_c0 + .5f;
    const float f_c1 = c1 - .5f * voxels_c1 + .5f;
    const float f_c2 = c2 - .5f * voxels_c2 + .5f;

    // map voxel coordinates to physical coordinates
    const float& X = f_c{{volume_axes.index(0)}};
    const float& Y = f_c{{volume_axes.index(1)}};
    const float& Z = f_c{{volume_axes.index(2)}};

    float acc[{{voxels_per_block[0]}}];
    for (int i = 0; i < {{voxels_per_block[0]}}; ++i)
        acc[i] = .0f;

    for (int j = start; j < end; ++j) {
        const float4& nU = params[j].numeratorU;
        const float4& nV = params[j].numeratorV;
        const float4& d  = params[j].denominator;

        // in starting position (X, Y, Z)
        float numU = nU.w + X * nU.x + Y * nU.y + Z * nU.z;
        float numV = nV.w + X * nV.x + Y * nV.y + Z * nV.z;
        float den  = d.w  + X * d.x  + Y * d.y  + Z * d.z;

        float r, U, V;
        for (int i = 0; i < {{voxels_per_block[0]}}; ++i) {
            r = __fdividef(1.f, den);
            U = numU * r;
            V = numV * r;
{% if texture == 'tex2DLayered' %}
{% set ax = ['j', 'V', 'U'] %}
            float val = tex2DLayered<float>(
                projTexture,
                {{ ax[projection_axes[2]] }},
                {{ ax[projection_axes[1]] }},
                {{ ax[projection_axes[0]] }});
{% elif texture == 'surf2DLayered' %}
{% set ax = ['j', 'V', 'U'] %}
            float val = tex2DLayered<float>(
                projTexture,
                {{ ax[projection_axes[2]] }},
                {{ ax[projection_axes[1]] }},
                {{ ax[projection_axes[0]] }});
{% elif texture == 'tex2D' %}
{% set ax = ['j', 'V', 'U'] %}
            float val = tex2D<float>(
                projTextures[{{ ax[projection_axes[0]] }}],
                {{ ax[projection_axes[2]] }},
                {{ ax[projection_axes[1]] }});
{% elif texture == 'tex3D' %}
{% set ax = ['j + .5f', 'V', 'U'] %}
            float val = tex3D<float>(
                projTexture,
                {{ ax[projection_axes[2]] }},
                {{ ax[projection_axes[1]] }},
                {{ ax[projection_axes[0]] }});
{% else %}
            float val=0.0f;
            printf("Implementation error: unknown texture type '{{ texture }}'.\n");
            return;
{% endif %}
            acc[i] += r * r * val;

            // Progress one unit in the world axis corresponding to the first
            // dimension of the data as laid out in memory.
            {% set slice_dir = ['x', 'y', 'z'][volume_axes[0]] %}
            numU += nU.{{slice_dir}};
            numV += nV.{{slice_dir}};
            den  += d.{{slice_dir}};
        }
    }

    // coalesced write to Z
    for (int i = 0; i < min({{voxels_per_block[0]}}, voxels_c0 - start_c0); ++i)
        volume[((start_c0 + i) * voxels_c1 + c1) * voxels_c2 + c2]
             += acc[i] * outputScale;
}