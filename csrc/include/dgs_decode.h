/* dgs_decode.h
 *
 * contains definitions for decoding gaussians from the DGS format
 */

#include "dgs_global.h"
#include "dgs_error.h"
#include "dgs_gaussians.h"
#include "dgs_format.h"

#ifndef DGS_DECODE_H
#define DGS_DECODE_H

//-------------------------------------------//

/**
 * decodes gaussians from the DGS format
 */
DGS_API DGSerror dgs_decode_from_file(const char* path, DGSgaussians* out, DGSmetadata* outMetadata);

/**
 * decodes gaussians from the DGS format
 */
DGS_API DGSerror dgs_decode_from_buffer(uint64_t size, const uint8_t* buf, DGSgaussians* out, DGSmetadata* outMetadata);

#endif //#ifndef DGS_DECODE_H