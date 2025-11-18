/* mgs_error.h
 *
 * contains the definition of the error enum
 */

#ifndef MGS_ERROR_H
#define MGS_ERROR_H

#include "mgs_global.h"

//-------------------------------------------//

/**
 * possible return codes from a function
 */
typedef enum MGSerror
{
	MGS_SUCCESS = 0,

	MGS_ERROR_INVALID_ARGUMENTS,
	MGS_ERROR_INVALID_INPUT,

	MGS_ERROR_OUT_OF_MEMORY,

	MGS_ERROR_FILE_OPEN,
	MGS_ERROR_FILE_CLOSE,
	MGS_ERROR_FILE_READ,
	MGS_ERROR_FILE_WRITE
} MGSerror;

//-------------------------------------------//

MGS_API const char* mgs_error_get_description(MGSerror error);

#endif //#ifndef MGS_ERROR_H
