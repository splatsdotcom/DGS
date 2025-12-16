/* dgs_error.h
 *
 * contains the definition of the error enum
 */

#ifndef DGS_ERROR_H
#define DGS_ERROR_H

#include "dgs_global.h"

//-------------------------------------------//

/**
 * possible return codes from a function
 */
typedef enum DGSerror
{
	DGS_SUCCESS = 0,

	DGS_ERROR_INVALID_ARGUMENTS,
	DGS_ERROR_INVALID_INPUT,

	DGS_ERROR_OUT_OF_MEMORY,

	DGS_ERROR_FILE_OPEN,
	DGS_ERROR_FILE_CLOSE,
	DGS_ERROR_FILE_READ,
	DGS_ERROR_FILE_WRITE
} DGSerror;

//-------------------------------------------//

DGS_API const char* dgs_error_get_description(DGSerror error);

#endif //#ifndef DGS_ERROR_H
