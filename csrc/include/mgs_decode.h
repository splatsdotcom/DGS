/* mgs_decode.h
 *
 * contains definitions for decoding gaussians from the MGS format
 */

#include "mgs_global.h"
#include "mgs_error.h"
#include "mgs_gaussians.h"
#include "mgs_format.h"

#ifndef MGS_DECODE_H
#define MGS_DECODE_H

//-------------------------------------------//

/**
 * decodes gaussians from the MGS format
 */
MGS_API MGSerror mgs_decode_from_file(const char* path, MGSgaussians* out, MGSmetadata* outMetadata);

/**
 * decodes gaussians from the MGS format
 */
MGS_API MGSerror mgs_decode_from_buffer(uint64_t size, const uint8_t* buf, MGSgaussians* out, MGSmetadata* outMetadata);

#endif //#ifndef MGS_DECODE_H