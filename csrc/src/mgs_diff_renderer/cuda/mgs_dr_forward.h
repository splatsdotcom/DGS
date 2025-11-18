/* mgs_dr_forward.h
 *
 * contains declarations for the forward rendering functions,
 * implemented in CUDA
 */

#ifndef MGS_DR_FORWARD_H
#define MGS_DR_FORWARD_H

#include <functional>
#include <stdint.h>
#include "mgs_dr_global.h"

typedef std::function<uint8_t* (uint64_t size)> MGSDRresizeFunc;

//-------------------------------------------//

uint32_t mgs_dr_forward_cuda(MGSDRsettings settings, MGSDRgaussians gaussians,
                             MGSDRresizeFunc createGeomBuf, MGSDRresizeFunc createBinningBuf, MGSDRresizeFunc createImageBuf,
                             float* outImg);

#endif //#ifndef MGS_DR_FOWARD_H