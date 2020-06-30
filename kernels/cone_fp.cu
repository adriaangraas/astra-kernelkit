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
//#include "astra/cuda/3d/util3d.h"
//#include "astra/cuda/3d/dims3d.h"
//typedef texture<float, 3, cudaReadModeElementType> texture3D;
//static texture3D gT_coneVolumeTexture;

extern "C" {
//    static const unsigned int g_anglesPerBlock = 4;

//    // thickness of the slices we're splitting the volume up into
//    static const unsigned int g_blockSlices = 4;
//    static const unsigned int g_detBlockU = 32;
//    static const unsigned int g_detBlockV = 32;

//    __constant__ float gC_SrcX[g_MaxAngles];
//    __constant__ float gC_SrcY[g_MaxAngles];
//    __constant__ float gC_SrcZ[g_MaxAngles];
//    __constant__ float gC_DetSX[g_MaxAngles];
//    __constant__ float gC_DetSY[g_MaxAngles];
//    __constant__ float gC_DetSZ[g_MaxAngles];
//    __constant__ float gC_DetUX[g_MaxAngles];
//    __constant__ float gC_DetUY[g_MaxAngles];
//    __constant__ float gC_DetUZ[g_MaxAngles];
//    __constant__ float gC_DetVX[g_MaxAngles];
//    __constant__ float gC_DetVY[g_MaxAngles];
//    __constant__ float gC_DetVZ[g_MaxAngles];

    // x=0, y=1, z=2
    struct DIR_X {
        __device__ float nSlices(const SDimensions3D &dims) const { return dims.iVolX; }
        __device__ float nDim1(const SDimensions3D &dims) const { return dims.iVolY; }
        __device__ float nDim2(const SDimensions3D &dims) const { return dims.iVolZ; }
        __device__ float c0(float x, float y, float z) const { return x; }
        __device__ float c1(float x, float y, float z) const { return y; }
        __device__ float c2(float x, float y, float z) const { return z; }
        __device__ float tex(float f0, float f1, float f2) const { return tex3D(gT_coneVolumeTexture, f0, f1, f2); }
        __device__ float x(float f0, float f1, float f2) const { return f0; }
        __device__ float y(float f0, float f1, float f2) const { return f1; }
        __device__ float z(float f0, float f1, float f2) const { return f2; }
    };

    // y=0, x=1, z=2
    struct DIR_Y {
        __device__ float nSlices(const SDimensions3D &dims) const { return dims.iVolY; }
        __device__ float nDim1(const SDimensions3D &dims) const { return dims.iVolX; }
        __device__ float nDim2(const SDimensions3D &dims) const { return dims.iVolZ; }
        __device__ float c0(float x, float y, float z) const { return y; }
        __device__ float c1(float x, float y, float z) const { return x; }
        __device__ float c2(float x, float y, float z) const { return z; }
        __device__ float tex(float f0, float f1, float f2) const { return tex3D(gT_coneVolumeTexture, f1, f0, f2); }
        __device__ float x(float f0, float f1, float f2) const { return f1; }
        __device__ float y(float f0, float f1, float f2) const { return f0; }
        __device__ float z(float f1, float f1, float f2) const { return f2; }
    };

    // z=0, x=1, y=2
    struct DIR_Z {
        __device__ float nSlices(const SDimensions3D &dims) const { return dims.iVolZ; }
        __device__ float nDim1(const SDimensions3D &dims) const { return dims.iVolX; }
        __device__ float nDim2(const SDimensions3D &dims) const { return dims.iVolY; }
        __device__ float c0(float x, float y, float z) const { return z; }
        __device__ float c1(float x, float y, float z) const { return x; }
        __device__ float c2(float x, float y, float z) const { return y; }
        __device__ float tex(float f0, float f1, float f2) const { return tex3D(gT_coneVolumeTexture, f1, f2, f0); }
        __device__ float x(float f0, float f1, float f2) const { return f1; }
        __device__ float y(float f0, float f1, float f2) const { return f2; }
        __device__ float z(float f0, float f1, float f2) const { return f0; }
    };

    struct SCALE_CUBE {
        float fOutputScale;
        __device__ float scale(float a1, float a2) const {
            return sqrt(a1 * a1 + a2 * a2 + 1.0f) * fOutputScale;
        }
    };

    struct SCALE_NONCUBE {
        float fScale1;
        float fScale2;
        float fOutputScale;

        __device__ float scale(float a1, float a2) const {
            return sqrt(a1 * a1 * fScale1 + a2 * a2 * fScale2 + 1.0f) * fOutputScale;
        }
    };

    // threadIdx: x = ??? detector  (u?)
    //            y = relative angle

    // blockIdx:  x = ??? detector  (u+v?)
    //            y = angle block

    __global__ void cone_FP_t(float *D_projData,
                              unsigned int projPitch,
                              unsigned int startSlice,
                              unsigned int startAngle,
                              unsigned int endAngle,
                              const SDimensions3D dims,
                              {{ type_scale }} sc) {
        {{ type_coord }} c;

        int angle = startAngle + blockIdx.y * {{ angles_per_block }} + threadIdx.y;

        if (angle >= endAngle)
            return;

        const float fSrcX = gC_SrcX[angle];
        const float fSrcY = gC_SrcY[angle];
        const float fSrcZ = gC_SrcZ[angle];
        const float fDetUX = gC_DetUX[angle];
        const float fDetUY = gC_DetUY[angle];
        const float fDetUZ = gC_DetUZ[angle];
        const float fDetVX = gC_DetVX[angle];
        const float fDetVY = gC_DetVY[angle];
        const float fDetVZ = gC_DetVZ[angle];
        const float fDetSX = gC_DetSX[angle] + 0.5f * fDetUX + 0.5f * fDetVX;
        const float fDetSY = gC_DetSY[angle] + 0.5f * fDetUY + 0.5f * fDetVY;
        const float fDetSZ = gC_DetSZ[angle] + 0.5f * fDetUZ + 0.5f * fDetVZ;

        const int detectorU =
            (blockIdx.x % ((dims.iProjU + {{ det_block_U }} - 1) / {{ det_block_U }})) * {{ det_block_U }} + threadIdx.x;

        const int startDetectorV =
            (blockIdx.x / ((dims.iProjU + {{ det_block_U }} - 1) / {{ det_block_U }})) * {{ det_block_V }};

        int endDetectorV = startDetectorV + {{ det_block_V}};

        if (endDetectorV > dims.iProjV)
            endDetectorV = dims.iProjV;

        int endSlice = startSlice + {{ block_slices }};
        if (endSlice > c.nSlices(dims))
            endSlice = c.nSlices(dims);


        for (int detectorV = startDetectorV; detectorV < endDetectorV; ++detectorV) {
            /* Trace ray from Src to (detectorU,detectorV) from
             * X = startSlice to X = endSlice
             */

            const float fDetX = fDetSX + detectorU * fDetUX + detectorV * fDetVX;
            const float fDetY = fDetSY + detectorU * fDetUY + detectorV * fDetVY;
            const float fDetZ = fDetSZ + detectorU * fDetUZ + detectorV * fDetVZ;

            /*        (x)   ( 1)       ( 0)
             * ray:   (y) = (ay) * x + (by)
             *        (z)   (az)       (bz)
             */

            const float a1 = (c.c1(fSrcX, fSrcY, fSrcZ) - c.c1(fDetX, fDetY, fDetZ)) /
                             (c.c0(fSrcX, fSrcY, fSrcZ) - c.c0(fDetX, fDetY, fDetZ));
            const float a2 = (c.c2(fSrcX, fSrcY, fSrcZ) - c.c2(fDetX, fDetY, fDetZ)) /
                             (c.c0(fSrcX, fSrcY, fSrcZ) - c.c0(fDetX, fDetY, fDetZ));
            const float b1 = c.c1(fSrcX, fSrcY, fSrcZ) - a1 * c.c0(fSrcX, fSrcY, fSrcZ);
            const float b2 = c.c2(fSrcX, fSrcY, fSrcZ) - a2 * c.c0(fSrcX, fSrcY, fSrcZ);

            const float fDistCorr = sc.scale(a1, a2);

            float fVal = 0.f;

            float f0 = startSlice + .5f;
            float f1 = a1 * (startSlice - .5f * c.nSlices(dims) + .5f) + b1 + .5f * c.nDim1(dims) - .5f + .5f;
            float f2 = a2 * (startSlice - .5f * c.nSlices(dims) + .5f) + b2 + .5f * c.nDim2(dims) - .5f + .5f;

            for (int s = startSlice; s < endSlice; ++s) {
                fVal += c.tex(f0, f1, f2);
                f0 += 1.f;
                f1 += a1;
                f2 += a2;
            }

            fVal *= fDistCorr;

            D_projData[(detectorV * dims.iProjAngles + angle) * projPitch + detectorU] += fVal;
        }
    }
}

