#define __FILENAME__ "mgs_encode.c"

#include "mgs_encode.h"
#include "mgs_log.h"
#include <stdio.h>

//-------------------------------------------//

MGSerror _mgs_fwrite(FILE* file, const void* data, size_t size);

//-------------------------------------------//

MGSerror mgs_encode(const MGSgaussians* g, const char* outputPath)
{
	MGSerror retval = MGS_SUCCESS;

	FILE* out = NULL;

	//open file:
	//---------------
	out = fopen(outputPath, "wb");
	if(!out)
	{
		MGS_LOG_ERROR("failed to open output file to write");
		retval = MGS_ERROR_FILE_OPEN;
		goto cleanup;
	}

	//write metadata:
	//---------------
	MGS_ERROR_PROPAGATE(_mgs_fwrite(out, &g->count   , sizeof(uint32_t)));
	MGS_ERROR_PROPAGATE(_mgs_fwrite(out, &g->dynamic , sizeof(mgs_bool_t)));

	MGS_ERROR_PROPAGATE(_mgs_fwrite(out, &g->shDegree, sizeof(uint32_t)));
	
	MGS_ERROR_PROPAGATE(_mgs_fwrite(out, &g->colorMin, sizeof(float)));
	MGS_ERROR_PROPAGATE(_mgs_fwrite(out, &g->colorMax, sizeof(float)));
	MGS_ERROR_PROPAGATE(_mgs_fwrite(out, &g->shMin   , sizeof(float)));
	MGS_ERROR_PROPAGATE(_mgs_fwrite(out, &g->shMax   , sizeof(float)));
	
	//write data:
	//---------------
	uint32_t numShCoeff = (g->shDegree + 1) * (g->shDegree + 1) - 1;

	MGS_ERROR_PROPAGATE(_mgs_fwrite(out, g->means, g->count * 4 * sizeof(float)));
	MGS_ERROR_PROPAGATE(_mgs_fwrite(out, g->covariances, g->count * 6 * sizeof(float)));
	MGS_ERROR_PROPAGATE(_mgs_fwrite(out, g->opacities, g->count * sizeof(uint8_t)));
	MGS_ERROR_PROPAGATE(_mgs_fwrite(out, g->colors, g->count * 3 * sizeof(uint16_t)));

	if(numShCoeff > 0)
	{
		MGS_ERROR_PROPAGATE(_mgs_fwrite(out, g->shs, g->count * numShCoeff * 3 * sizeof(uint8_t)));
	}

	if(g->dynamic)
	{
		MGS_ERROR_PROPAGATE(_mgs_fwrite(out, g->velocities, g->count * 4 * sizeof(float)));
	}

	//close file:
	//---------------
	if(fclose(out) != 0)
	{
		MGS_LOG_ERROR("failed to close output file after writing");
		retval = MGS_ERROR_FILE_CLOSE;
		goto cleanup;	
	}

	//cleanup + return:
	//---------------
cleanup:
	if(out)
		fclose(out);

	return retval;
}

MGSerror _mgs_fwrite(FILE* file, const void* data, size_t size)
{
	if(fwrite(data, size, 1, file) < 1)
	{
		MGS_LOG_ERROR("failed to write to file");
		return MGS_ERROR_FILE_WRITE;
	}

	return MGS_SUCCESS;
}