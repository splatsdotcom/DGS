/* mgs_encode.h
 *
 * contains definitions for encoding gaussians to the MGS format
 */

#include "mgs_global.h"
#include "mgs_error.h"
#include "mgs_gaussians.h"
#include "mgs_format.h"

#ifndef MGS_ENCODE_H
#define MGS_ENCODE_H

//-------------------------------------------//

/**
 * encodes gaussians to the MGS format
 */
MGS_API MGSerror mgs_encode(const MGSgaussians* gaussians, MGSmetadata metadata, const char* outputPath);

#endif //#ifndef MGS_ENCODE_H
