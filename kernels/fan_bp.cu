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
    static const unsigned int g_blockSliceSize = 32;
    static const unsigned int g_blockSlices = 16;

    /**
     * Backprojection for fan geometry with flat detector
     * Note this is just a PoC and not likely to hit the performance of ASTRA Toolbox.
     *
     * @todo
     *  - supersampling
     *  - SART version / FBPWEIGHT
     *  - transfer DevFanParams to constant memory
     */
    __global__ void
    fan_bp(
        cudaTextureObject_t projTexture,
        float* volume,
        float* params,                     // in shared memory (too slow?)
        int startAngle,
        int nrAngles,
        int volWidth,                      // volume (txt memory) width
        int volHeight,                     // volume (txt memory) height
        float outputScale
    ) {
        const int relX = threadIdx.x;
        const int relY = threadIdx.y;

        int endAngle = startAngle + g_anglesPerBlock;

        if (endAngle > nrAngles)
            endAngle = nrAngles;

        const int blockX = blockIdx.x * g_blockSlices + relX;
        const int blockY = blockIdx.y * g_blockSliceSize + relY;

        if (blockX >= volWidth || blockY >= volHeight)
            return;

        const float X = blockX - .5f * volWidth + .5f;
        const float Y = - (blockY - 0.5f * volHeight + 0.5f);

        float val = .0f;
        float a = startAngle + .5f;

        for (int angle = startAngle * 6; angle < endAngle * 6; angle += 6)
        {
            const float numC = params[angle];
            const float numX = params[angle +1];
            const float numY = params[angle + 2];
            const float denX = params[angle + 4];
            const float denY = params[angle + 5];

            const float num = numC + numX * X + numY * Y;
            const float den = params[angle + 3] + denX * X + denY * Y;

            // Scale factor is the approximate number of rays traversing this pixel,
            // given by the inverse size of a detector pixel scaled by the magnification
            // factor of this pixel.
            // Magnification factor is || u (d-s) || / || u (x-s) ||
            const float r = __fdividef(1.f, den);
            const float T = num * r;

            val += tex2D<float>(projTexture, T, a) * r;
            a += 1.f;
        }

        volume[blockY * volWidth + blockX] += val * outputScale;
    }
}
