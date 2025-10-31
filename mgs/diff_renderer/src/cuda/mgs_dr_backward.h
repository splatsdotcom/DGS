/* mgs_dr_backward.h
 *
 * contains declataions for the backward rendering pass,
 * implemented in CUDA
 */

#ifndef MGS_DR_BACKWARD_H
#define MGS_DR_BACKWARD_H

#include <stdint.h>
#include "mgs_dr_global.h"

//-------------------------------------------//

void mgs_dr_backward_cuda(MGSDRsettings settings, const float* dLdImage, MGSDRgaussians gaussians,
                          uint32_t numRendered, const uint8_t* geomBufsMem, const uint8_t* binningBufsMem, const uint8_t* imageBufsMem,
                          MGSDRgaussians outDLdGaussians);

#endif //#ifndef MGS_DR_BACKWARD_H