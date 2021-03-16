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

static const unsigned int volBlockX = {{ vol_block_x }};
static const unsigned int volBlockY = {{ vol_block_y }};
static const unsigned int volBlockZ = {{ vol_block_z }};

struct Params {
	float4 numeratorU;
	float4 numeratorV;
	float4 denominator;
};

__constant__ Params params[{{ max_angles }}];

__device__ __forceinline__
void volumeAdd(
    float * vol,
    int voxelsX, int voxelsY, int voxelsZ,
    int x, int y, int z,
    float val
) {
    vol[  z * voxelsX * voxelsY
        + y * voxelsX
        + x] += val;

//    vol[  z * voxelsZ * voxelsY
//        + y * voxelsZ
//        + x] += val;
}

template<bool FDKWEIGHT>
__global__
void cone_bp(
    texture3D projTexture,
    float * volume,
    int startAngle,
    int nrAngles,
    int voxelsX,
    int voxelsY,
    int voxelsZ,
    float outputScale
) {
	int endAngle = startAngle + {{ angles_per_block }};
	if (endAngle > nrAngles)
		endAngle = nrAngles;

    // TODO(Adriaan): why not use blockDim?
    const int n = ((voxelsX + volBlockX - 1) / volBlockX);
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
	float Z[{{ vol_block_z }}];
	for(int i=0; i < {{ vol_block_z }}; i++)
		Z[i] = .0f;

    // scope hints the compiler to clean up variables?
	{
		float angle = startAngle + .5f;
		for (int a = startAngle; a < endAngle; ++a, angle += 1.f) {
			float4 nU = params[a].numeratorU;
			float4 nV = params[a].numeratorV;
			float4 d  = params[a].denominator;

			float numU = nU.w + fX * nU.x + fY * nU.y + fZ * nU.z;
			float numV = nV.w + fX * nV.x + fY * nV.y + fZ * nV.z;
			float den  = (FDKWEIGHT ? 1.f : d.w) + fX * d.x + fY * d.y + fZ * d.z;

			float r, U, V;
			for (int z = 0; z < {{ vol_block_z }}; z++) {
				r = __fdividef(1.f, den);

				U = numU * r;
				V = numV * r;
				float val = tex3D(projTexture, U, V, angle);
				Z[z] += r * r * val;

				numU += nU.z;
				numV += nV.z;
				den  += d.z;
			}
		}
	}

    // make sure to write inside volume
	int endZ = {{ vol_block_z }};
	if (endZ > voxelsZ - startZ)
		endZ = voxelsZ - startZ;

    // write Z to volume synchronously
	for(int i=0; i < endZ; i++)
	    volumeAdd(volume,
	              voxelsX, voxelsY, voxelsZ,
	              X, Y, startZ + i,
	              Z[i] * outputScale);
}