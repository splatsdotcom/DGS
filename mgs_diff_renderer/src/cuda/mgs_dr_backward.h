/* mgs_dr_backward.h
 *
 * contains declataions for the backward rendering pass,
 * implemented in CUDA
 */

#ifndef MGS_DR_BACKWARD_H
#define MGS_DR_BACKWARD_H

#include <stdint.h>

//-------------------------------------------//

void mgs_dr_backward_cuda(uint32_t width, uint32_t height, const float* dLdImage, const float* view, const float* proj, float focalX, float focalY,
                          uint32_t numGaussians, const float* means, const float* scales, const float* rotations, const float* opacities, const float* colors, const float* harmonics,
                          uint32_t numRendered, const uint8_t* geomBufsMem, const uint8_t* binningBufsMem, const uint8_t* imageBufsMem,
                          float* outDLdMeans, float* outDLdScales, float* outDLdRotations, float* outDLdOpacities, float* outDLdColors, float* outDLdHarmonics,
                          bool debug);

#endif //#ifndef MGS_DR_BACKWARD_H