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
static const unsigned int volBlockX = {{ nr_vxls_block_x }};
static const unsigned int volBlockY = {{ nr_vxls_block_y }};
static const unsigned int volBlockZ = {{ nr_vxls_block_z }};

struct Params {
	float4 numeratorU;
	float4 numeratorV;
	float4 denominator;
};

__constant__ Params params[{{ nr_projs_global }}];

__global__ void cone_bp(
{% if texture3D %}
    cudaTextureObject_t projTexture,
{% else %}
    cudaTextureObject_t * projTextures,
{% endif %}
    float * volume,
    int start,
    int nrProjections,
    int voxelsX,
    int voxelsY,
    int voxelsZ,
    float outputScale
) {
	int end = start + {{ nr_projs_block }};
	if (end > nrProjections)
		end = nrProjections;

    const int n = (voxelsX + volBlockX - 1) / volBlockX;
	const int X = blockIdx.x % n * volBlockX + threadIdx.x;
	const int Y = blockIdx.x / n * volBlockY + threadIdx.y;

	if (X >= voxelsX || Y >= voxelsY)
		return;

	const int startZ = blockIdx.y * volBlockZ;

	// shift voxel coordinates to center, add .5 for texture coord
	const float fX = X - .5f * voxelsX + .5f;
	const float fY = Y - .5f * voxelsY + .5f;
	const float fZ = startZ - .5f * voxelsZ + .5f;

    // init Z to zero
	float Z[{{ nr_vxls_block_z }}];
	for (int i = 0; i < {{ nr_vxls_block_z }}; ++i)
		Z[i] = .0f;

    // scope hints the compiler to clean up variables?
	{
		for (int j = start; j < end; ++j) {
			const float4& nU = params[j].numeratorU;
			const float4& nV = params[j].numeratorV;
			const float4& d  = params[j].denominator;

			float numU = nU.w + fX * nU.x + fY * nU.y + fZ * nU.z;
			float numV = nV.w + fX * nV.x + fY * nV.y + fZ * nV.z;
			float den  = d.w  + fX * d.x  + fY * d.y  + fZ * d.z;

			float r, U, V;
			for (int i = 0; i < {{ nr_vxls_block_z }}; ++i) {
				r = __fdividef(1.f, den);
				U = numU * r;
				V = numV * r;
{% if texture3D %}
				float val = tex3D<float>(projTexture, U, V, j + .5f);
{% else %}
                float val = tex2D<float>(projTextures[j], U, V);
{% endif %}
				Z[i] += r * r * val;

				numU += nU.z;
				numV += nV.z;
				den  += d.z;
			}
		}
	}

    // make sure to write inside volume
	int endZ = {{ nr_vxls_block_z }};
	if (endZ > voxelsZ - startZ)
		endZ = voxelsZ - startZ;

    // write Z to volume synchronously
    // Adriaan: an optimized write pattern does not seem to make a lot of effect
	for (int i = 0; i < endZ; ++i)
	    volume[  X * voxelsY * voxelsZ
               + Y * voxelsZ
               + startZ + i] += Z[i] * outputScale;
}