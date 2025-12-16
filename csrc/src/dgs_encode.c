#define __FILENAME__ "dgs_encode.c"

#include "dgs_encode.h"
#include "dgs_log.h"
#include <stdio.h>

//-------------------------------------------//

static DGSerror _dgs_fwrite(FILE* file, const void* data, size_t size);

//-------------------------------------------//

DGSerror dgs_encode(const DGSgaussians* g, DGSmetadata metadata, const char* outputPath)
{
	DGSerror retval = DGS_SUCCESS;

	FILE* out = NULL;

	//open file:
	//---------------
	out = fopen(outputPath, "wb");
	if(!out)
	{
		DGS_LOG_ERROR("failed to open output file to write");
		retval = DGS_ERROR_FILE_OPEN;
		goto cleanup;
	}

	//write file header + metadata:
	//---------------
	DGSfileHeader header;
	header.magicWord = DGS_MAGIC_WORD;
	header.version = DGS_VERSION;

	DGS_ERROR_PROPAGATE(_dgs_fwrite(out, &header, sizeof(DGSfileHeader)));
	DGS_ERROR_PROPAGATE(_dgs_fwrite(out, &metadata, sizeof(DGSmetadata)));

	//write gaussian metadata:
	//---------------
	DGS_ERROR_PROPAGATE(_dgs_fwrite(out, &g->count   , sizeof(uint32_t)));
	DGS_ERROR_PROPAGATE(_dgs_fwrite(out, &g->dynamic , sizeof(dgs_bool_t)));

	DGS_ERROR_PROPAGATE(_dgs_fwrite(out, &g->shDegree, sizeof(uint32_t)));
	
	DGS_ERROR_PROPAGATE(_dgs_fwrite(out, &g->colorMin, sizeof(float)));
	DGS_ERROR_PROPAGATE(_dgs_fwrite(out, &g->colorMax, sizeof(float)));
	DGS_ERROR_PROPAGATE(_dgs_fwrite(out, &g->shMin   , sizeof(float)));
	DGS_ERROR_PROPAGATE(_dgs_fwrite(out, &g->shMax   , sizeof(float)));
	
	//write gaussian data:
	//---------------
	uint32_t numShCoeff = (g->shDegree + 1) * (g->shDegree + 1) - 1;

	DGS_ERROR_PROPAGATE(_dgs_fwrite(out, g->means, g->count * 4 * sizeof(float)));
	DGS_ERROR_PROPAGATE(_dgs_fwrite(out, g->covariances, g->count * 6 * sizeof(float)));
	DGS_ERROR_PROPAGATE(_dgs_fwrite(out, g->opacities, g->count * sizeof(uint8_t)));
	DGS_ERROR_PROPAGATE(_dgs_fwrite(out, g->colors, g->count * 3 * sizeof(uint16_t)));

	if(numShCoeff > 0)
	{
		DGS_ERROR_PROPAGATE(_dgs_fwrite(out, g->shs, g->count * numShCoeff * 3 * sizeof(uint8_t)));
	}

	if(g->dynamic)
	{
		DGS_ERROR_PROPAGATE(_dgs_fwrite(out, g->velocities, g->count * 4 * sizeof(float)));
	}

	//close file:
	//---------------
	if(fclose(out) != 0)
	{
		DGS_LOG_ERROR("failed to close output file after writing");
		retval = DGS_ERROR_FILE_CLOSE;
		goto cleanup;	
	}
	out = NULL;

	//cleanup + return:
	//---------------
cleanup:
	if(out)
		fclose(out);

	return retval;
}

static DGSerror _dgs_fwrite(FILE* file, const void* data, size_t size)
{
	if(fwrite(data, size, 1, file) < 1)
	{
		DGS_LOG_ERROR("failed to write to file");
		return DGS_ERROR_FILE_WRITE;
	}

	return DGS_SUCCESS;
}