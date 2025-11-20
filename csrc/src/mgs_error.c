#define __FILENAME__ "mgs_error.c"

#include "mgs_error.h"

//-------------------------------------------//

const char* mgs_error_get_description(MGSerror error)
{
	switch(error)
	{
	case MGS_SUCCESS:
		return "success";

	case MGS_ERROR_INVALID_ARGUMENTS:
		return "invalid arguments (bad function call)";
	case MGS_ERROR_INVALID_INPUT:
		return "invalid input (provided memory/file input was in an invalid format)";

	case MGS_ERROR_OUT_OF_MEMORY:
		return "out of memory (MGS_MALLOC/MGS_REALLOC returned NULL)";

	case MGS_ERROR_FILE_OPEN:
		return "failed to open a file (fopen returned NULL)";
	case MGS_ERROR_FILE_CLOSE:
		return "failed to close a file (fclose returned EOF)";
	case MGS_ERROR_FILE_READ:
		return "failed to read from a file (fread read fewer bytes than requested)";
	case MGS_ERROR_FILE_WRITE:
		return "failed to write to a file (fwrite wrote fewer bytes than requested)";
	default:
		return "unknown error";
	}
}