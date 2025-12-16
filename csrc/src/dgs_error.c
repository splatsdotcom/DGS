#define __FILENAME__ "dgs_error.c"

#include "dgs_error.h"

//-------------------------------------------//

const char* dgs_error_get_description(DGSerror error)
{
	switch(error)
	{
	case DGS_SUCCESS:
		return "success";

	case DGS_ERROR_INVALID_ARGUMENTS:
		return "invalid arguments (bad function call)";
	case DGS_ERROR_INVALID_INPUT:
		return "invalid input (provided memory/file input was in an invalid format)";

	case DGS_ERROR_OUT_OF_MEMORY:
		return "out of memory (DGS_MALLOC/DGS_REALLOC returned NULL)";

	case DGS_ERROR_FILE_OPEN:
		return "failed to open a file (fopen returned NULL)";
	case DGS_ERROR_FILE_CLOSE:
		return "failed to close a file (fclose returned EOF)";
	case DGS_ERROR_FILE_READ:
		return "failed to read from a file (fread read fewer bytes than requested)";
	case DGS_ERROR_FILE_WRITE:
		return "failed to write to a file (fwrite wrote fewer bytes than requested)";
	default:
		return "unknown error";
	}
}