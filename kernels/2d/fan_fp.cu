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
    __constant__ float csrcX[{{ max_angles }}];
    __constant__ float csrcY[{{ max_angles }}];
    __constant__ float cdetSX[{{ max_angles }}];
    __constant__ float cdetSY[{{ max_angles }}];
    __constant__ float cdetUX[{{ max_angles }}];
    __constant__ float cdetUY[{{ max_angles }}];

    /**
     * Forward projection for fan geometry with flat detector
     *
     * Adriaan: I'm waiting for C++17 to be supported in CUDA, then I can template bool horizontalMode, and use
     *          a "constexpr if" on the conditions. Until then I'll use Jinja templates!
     * Adriaan: Further TODO's on this kernel are
     * - pitched memory support is not (yet, or never) in CuPy, how much are we missing out?
     * - from ASTRA Toolbox: for very small sizes (roughly <=512x128) with few angles (<=180)
     *                       not using an array is more efficient.
     * - supersampling
     * - support output scaling in a templated fashion
     */
    __global__ void
    fan_fp(cudaTextureObject_t volumeTexture,  // volume in texture memory
           float * projections,                // list of angles to iterate
           unsigned int startSlice,            // ?
           unsigned int startAngle,            // global start index in float* angle
           unsigned int endAngle,              // global end index in float* angle
           int projDets,                       // number of pixels on the detector
           int raysPerDet,                     // ?
           int volWidth,                       // volume (txt memory) width
           int volHeight,                      // volume (txt memory) height
           float outputScale
    ) {
        const int relDet = threadIdx.x;
        const int relAngle = threadIdx.y;
        const int angle = startAngle + blockIdx.x * {{ angles_per_block }} + relAngle;

        if (angle >= endAngle)
            return;

        const int pixel = blockIdx.y * {{ det_block_size }} + relDet;

        if (pixel < 0 || pixel >= projDets)
            return;

        const float srcX = csrcX[angle];
        const float srcY = csrcY[angle];
        const float detSX = cdetSX[angle];
        const float detSY = cdetSY[angle];
        const float detUX = cdetUX[angle];
        const float detUY = cdetUY[angle];

        const float fdx = fabsf(detSX + pixel * detUX + 0.5f - srcX);
        const float fdy = fabsf(detSY + pixel * detUY + 0.5f - srcY);

        {% if mode_horizontal is sameas true %}
        if (fdy > fdx)
            return;
        {% else %}
        if (fdy <= fdx)
            return;
        {% endif %}

        float val = 0.0;
        for (int subT = 0; subT < raysPerDet; ++subT) {
            const float det = pixel + (0.5 + subT) / raysPerDet;
            const float detX = detSX + det * detUX;
            const float detY = detSY + det * detUY;

            // ray: y = alpha * x + beta
            float alpha, beta;

            {% if mode_horizontal is sameas true %}
            alpha = (srcY - detY) / (srcX - detX);
            beta = srcY - alpha * srcX;
            {% else %}
            alpha = (srcX - detX) / (srcY - detY);
            beta = srcX - alpha * srcY;
            {% endif %}

            const float distCorr = sqrt(alpha * alpha + 1) * outputScale / raysPerDet;

            // intersect ray with first slice
            float X, Y;
            {% if mode_horizontal is sameas true %}
            Y = -alpha * (startSlice - .5 * volWidth + .5) - beta + .5 * volHeight;
            X = startSlice + .5;
            {% else %}
            X = -alpha * (startSlice - .5 * volHeight + .5) + beta + .5 * volWidth;
            Y = startSlice + .5;
            {% endif %}

            int endSlice = min(startSlice + {{ block_slices }},
                {% if mode_horizontal is sameas true %} volWidth {% else %} volHeight {% endif %});

            float V = 0;
            for (int slice = startSlice; slice < endSlice; ++slice) {
                V += tex2D<float>(volumeTexture, X, Y);

                {% if mode_horizontal is sameas true %}
                Y -= alpha;
                X += 1;
                {% else %}
                X -= alpha;
                Y += 1;
                {% endif %}
            }

           val += V * distCorr;
        }

        projections[angle * projDets + pixel] += val;
    }
}
