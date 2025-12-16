/* dgs_encode.h
 *
 * contains definitions for encoding gaussians to the DGS format
 */

#include "dgs_global.h"
#include "dgs_error.h"
#include "dgs_gaussians.h"
#include "dgs_format.h"

#ifndef DGS_ENCODE_H
#define DGS_ENCODE_H

//-------------------------------------------//

/**
 * encodes gaussians to the DGS format
 */
DGS_API DGSerror dgs_encode(const DGSgaussians* gaussians, DGSmetadata metadata, const char* outputPath);

#endif //#ifndef DGS_ENCODE_H
