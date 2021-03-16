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
    struct Params {
        float numC;
        float numX;
        float numY;
        float denC;
        float denX;
        float denY;
    };
    __constant__ Params params[{{ max_angles }}];

    /**
     * Backprojection for fan geometry with flat detector
     *
     * TODO:
     *  - supersampling
     *  - SART version / FBPWEIGHT
     */
    __global__ void
    fan_bp(
        cudaTextureObject_t projTexture,
        float* volume,
        int startAngle,
        int nrAngles,
        int volWidth,                      // volume (txt memory) width
        int volHeight,                     // volume (txt memory) height
        float outputScale
    ) {
        const int relX = threadIdx.x;
        const int relY = threadIdx.y;

        int endAngle = startAngle + {{ angles_per_block }};

        if (endAngle > nrAngles)
            endAngle = nrAngles;

        const int blockX = blockIdx.x * {{ block_slices }} + relX;
        const int blockY = blockIdx.y * {{ block_slice_size }} + relY;

        if (blockX >= volWidth || blockY >= volHeight)
            return;

        const float X = blockX - .5f * volWidth + .5f;
        const float Y = - (blockY - .5f * volHeight + .5f);

        float val = .0f;
        float a = startAngle + .5f;

        for (int angle = startAngle; angle < endAngle; ++angle, a += 1.f)
        {
            const Params p = params[angle];
            const float num = p.numC + p.numX * X + p.numY * Y;
            const float den = p.denC + p.denX * X + p.denY * Y;

            // Scale factor is the approximate number of rays traversing this pixel,
            // given by the inverse size of a detector pixel scaled by the magnification
            // factor of this pixel.
            // Magnification factor is || u (d-s) || / || u (x-s) ||

            // inverse size detector pixel
            const float r = __fdividef(1.f, den);
            // scaled by magnification factor of the pixel
            const float T = num * r;

            val += tex2D<float>(projTexture, T, a) * r;
        }

        volume[blockY * volWidth + blockX] += val * outputScale;
    }
}
