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
    static const unsigned int g_anglesPerBlock = 16;
    static const unsigned int g_detBlockSize = 32;
    static const unsigned int g_blockSlices = 64;

    /**
     * Forward projection for fan geometry with flat detector
     *
     * Adriaan: I'm waiting for C++17 to be supported in CUDA, then I can template bool horizontalMode, and use
     *          a "constexpr if" on the conditions. Until then I'll accept slower runtimes.
     * Adriaan: Further @todo on this kernel are
     * - pitched memory support is not (yet, or never) in CuPy, how much are we missing out?
     * - constant memory is in CuPy and we should use it here for the angles
     * - from ASTRA Toolbox: for very small sizes (roughly <=512x128) with few angles (<=180)
     *                       not using an array is more efficient.
     * - supersampling
     * - support output scaling in a templated fashion
     */
    __global__ void
    fan_fp(cudaTextureObject_t volumeTexture,  // volume in texture memory
           float * projections,                // list of angles to iterate
           float * angles,                     // angle list in shared memory (too slow?)
           unsigned int startSlice,            // ?
           unsigned int startAngle,            // global start index in float* angle
           unsigned int endAngle,              // global end index in float* angle
           int projDets,                       // number of pixels on the detector
           int raysPerDet,                     // ?
           int volWidth,                       // volume (txt memory) width
           int volHeight,                      // volume (txt memory) height
           bool modeHorizontal,                // how to compute the ray paths
           float outputScale
           ) {
        const int relDet = threadIdx.x;
        const int relAngle = threadIdx.y;
        const int angle = startAngle
                          + blockIdx.x * g_anglesPerBlock
                          + relAngle;

        if (angle >= endAngle)
            return;


        const int pixel = blockIdx.y * g_detBlockSize + relDet;

        if (pixel < 0 || pixel >= projDets)
            return;

        const float detSX = angles[angle*6];
        const float detSY = angles[angle*6+1];
        const float detUX = angles[angle*6+2];
        const float detUY = angles[angle*6+3];
        const float srcX = angles[angle*6+4];
        const float srcY = angles[angle*6+5];

        const float fdx = fabsf(detSX + pixel * detUX + 0.5f - srcX);
        const float fdy = fabsf(detSY + pixel * detUY + 0.5f - srcY);

        if (modeHorizontal) {
            if (fdy > fdx)
                return;
        } else {
            if (fdy <= fdx)
                return;
        }

        float val = 0.0;
        for (int subT = 0; subT < raysPerDet; ++subT) {
            const float det = pixel + (0.5 + subT) / raysPerDet;
            const float detX = detSX + det * detUX;
            const float detY = detSY + det * detUY;

            // ray: y = alpha * x + beta
            float alpha, beta;
            if (modeHorizontal) {
                alpha = (srcY - detY) / (srcX - detX);
                beta = srcY - alpha * srcX;
            } else {
                alpha = (srcX - detX) / (srcY - detY);
                beta = srcX - alpha * srcY;
            }

            float distCorr = sqrt(alpha * alpha + 1) * outputScale / raysPerDet;

            // intersect ray with first slice
            float X, Y;
            if (modeHorizontal) {
                Y = -alpha * (startSlice - .5 * volWidth + .5) - beta + .5 * volHeight;
                X = startSlice + .5;
            } else {
                X = -alpha * (startSlice - .5 * volHeight + .5) + beta + .5 * volWidth;
                Y = startSlice + .5;
            }

            int endSlice = min(startSlice + g_blockSlices, modeHorizontal ? volWidth : volHeight);

            float V = 0;
            for (int slice = startSlice; slice < endSlice; ++slice) {
                V += tex2D<float>(volumeTexture, X, Y);

                if (modeHorizontal) {
                    Y -= alpha;
                    X += 1;
                } else {
                    X -= alpha;
                    Y += 1;
                }
            }

           val += V * distCorr;
        }

        projections[angle * projDets + pixel] += val;
    }
}
