/* mgs_dr_forward.h
 *
 * contains declarations for the forward rendering functions,
 * implemented in CUDA
 */

#ifndef MGS_DR_FORWARD_H
#define MGS_DR_FORWARD_H

#include <functional>
#include <stdint.h>

//-------------------------------------------//

uint32_t mgs_dr_forward_cuda(uint32_t outWidth, uint32_t outHeight, float* outImg, const float* view, const float* proj, float focalX, float focalY,
                             uint32_t numGaussians, const float* means, const float* scales, const float* rotations, const float* opacities, const float* colors, const float* harmonics,
                             std::function<uint8_t* (uint64_t size)> createGeomBuf, std::function<uint8_t* (uint64_t size)> createBinningBuf, std::function<uint8_t* (uint64_t size)> createImageBuf);

#endif //#ifndef MGS_DR_FOWARD_H