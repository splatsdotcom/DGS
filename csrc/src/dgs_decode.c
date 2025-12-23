#define __FILENAME__ "dgs_decode.c"

#include "dgs_decode.h"
#include "dgs_log.h"
#include <stdio.h>

//-------------------------------------------//

typedef struct DGSreader
{
	dgs_bool_t isFile;
	
	union
	{
		FILE* file;

		struct
		{
			uint64_t size;
			uint64_t pos;
			const uint8_t* buf;
		} buf;
	};
} DGSreader;

//-------------------------------------------//

static DGSerror _dgs_decode(DGSreader* reader, DGSgaussians* out, DGSmetadata* outMetadata);

static DGSerror _dgs_read(DGSreader* reader, void* data, uint64_t size);

//-------------------------------------------//

DGSerror dgs_decode_from_file(const char* path, DGSgaussians* out, DGSmetadata* outMetadata)
{
	DGSerror retval = DGS_SUCCESS;

	FILE* in = NULL;
	DGS_STRUCTURE_CLEAR(out);

	//open file:
	//---------------
	in = fopen(path, "rb");
	if(!in)
	{
		DGS_LOG_ERROR("failed to open input file to read");
		retval = DGS_ERROR_FILE_OPEN;
		goto cleanup;
	}	

	//decode:
	//---------------
	DGSreader reader;
	reader.isFile = DGS_TRUE;
	reader.file = in;

	DGS_ERROR_PROPAGATE(_dgs_decode(&reader, out, outMetadata));

	//cleanup + return:
	//---------------
cleanup:
	if(retval != DGS_SUCCESS)
		dgs_gaussians_free(out);
	if(in)
		fclose(in);

	return retval;
}

DGSerror dgs_decode_from_buffer(uint64_t size, const uint8_t* buf, DGSgaussians* out, DGSmetadata* outMetadata)
{
	DGSerror retval = DGS_SUCCESS;

	DGS_STRUCTURE_CLEAR(out);

	//decode:
	//---------------
	DGSreader reader;
	reader.isFile = DGS_FALSE;
	reader.buf.size = size;
	reader.buf.pos = 0;
	reader.buf.buf = buf;

	DGS_ERROR_PROPAGATE(_dgs_decode(&reader, out, outMetadata));

	//cleanup + return:
	//---------------
cleanup:
	if(retval != DGS_SUCCESS)
		dgs_gaussians_free(out);

	return retval;
}

static DGSerror _dgs_decode(DGSreader* reader, DGSgaussians* out, DGSmetadata* outMetadata)
{
	DGSerror retval = DGS_SUCCESS;

	//file header + metadata:
	//---------------
	DGSfileHeader header;

	DGS_ERROR_PROPAGATE(_dgs_read(reader, &header, sizeof(DGSfileHeader)));
	DGS_ERROR_PROPAGATE(_dgs_read(reader, outMetadata, sizeof(DGSmetadata)));

	//validate file header + metadata:
	//---------------
	if(header.magicWord != DGS_MAGIC_WORD)
	{
		DGS_LOG_ERROR("mismatched magic word");
		retval = DGS_ERROR_INVALID_INPUT;
		goto cleanup;
	}

	if(header.version < DGS_MAKE_VERSION(1, 1, 0) || header.version >= DGS_MAKE_VERSION(1, 2, 0))
	{
		DGS_LOG_ERROR("mismatched version! expected 1.1.x");
		retval = DGS_ERROR_INVALID_INPUT;
		goto cleanup;
	}

	if(outMetadata->duration < 0.0f)
		DGS_LOG_WARNING("negative duration encountered in metadata");

	//read gaussian properties:
	//---------------
	uint32_t count;
	dgs_bool_t dynamic;
	uint32_t shDegree;

	float scaleMin, scaleMax;
	float colorMin, colorMax;
	float shMin, shMax;

	DGS_ERROR_PROPAGATE(_dgs_read(reader, &count   , sizeof(uint32_t)));
	DGS_ERROR_PROPAGATE(_dgs_read(reader, &dynamic , sizeof(dgs_bool_t)));

	DGS_ERROR_PROPAGATE(_dgs_read(reader, &shDegree, sizeof(uint32_t)));
	
	DGS_ERROR_PROPAGATE(_dgs_read(reader, &scaleMin, sizeof(float)));
	DGS_ERROR_PROPAGATE(_dgs_read(reader, &scaleMax, sizeof(float)));
	DGS_ERROR_PROPAGATE(_dgs_read(reader, &colorMin, sizeof(float)));
	DGS_ERROR_PROPAGATE(_dgs_read(reader, &colorMax, sizeof(float)));
	DGS_ERROR_PROPAGATE(_dgs_read(reader, &shMin   , sizeof(float)));
	DGS_ERROR_PROPAGATE(_dgs_read(reader, &shMax   , sizeof(float)));

	//validate gaussian properties:
	//---------------
	if(count == 0)
	{
		DGS_LOG_ERROR("file contains 0 gaussians");
		retval = DGS_ERROR_INVALID_INPUT;
		goto cleanup;
	}

	if(shDegree > DGS_GAUSSIANS_MAX_SH_DEGREE)
	{
		DGS_LOG_ERROR("out of bounds sh degree");
		retval = DGS_ERROR_INVALID_INPUT;
		goto cleanup;
	}

	if(scaleMin > scaleMax)
	{
		DGS_LOG_ERROR("invalid scale normalization coefficients");
		retval = DGS_ERROR_INVALID_INPUT;
		goto cleanup;
	}

	if(colorMin > colorMax)
	{
		DGS_LOG_ERROR("invalid color normalization coefficients");
		retval = DGS_ERROR_INVALID_INPUT;
		goto cleanup;
	}

	if(shDegree > 0 && shMin > shMax)
	{
		DGS_LOG_ERROR("invalid sh normalization coefficients");
		retval = DGS_ERROR_INVALID_INPUT;
		goto cleanup;
	}
	
	//allocate gaussians:
	//---------------
	DGS_ERROR_PROPAGATE(dgs_gaussians_allocate(
		count, shDegree, dynamic, out
	));

	out->scaleMin = scaleMin;
	out->scaleMax = scaleMax;
	out->colorMin = colorMin;
	out->colorMax = colorMax;
	out->shMin = shMin;
	out->shMax = shMax;

	//read gaussian data:
	//---------------
	uint32_t numShCoeff = (shDegree + 1) * (shDegree + 1) - 1;

	DGS_ERROR_PROPAGATE(_dgs_read(reader, out->means, count * 4 * sizeof(float)));
	DGS_ERROR_PROPAGATE(_dgs_read(reader, out->scales, count * 3 * sizeof(uint16_t)));
	DGS_ERROR_PROPAGATE(_dgs_read(reader, out->rotations, count * 3 * sizeof(uint16_t)));
	DGS_ERROR_PROPAGATE(_dgs_read(reader, out->opacities, count * sizeof(uint8_t)));
	DGS_ERROR_PROPAGATE(_dgs_read(reader, out->colors, count * 3 * sizeof(uint16_t)));

	if(numShCoeff > 0)
	{
		DGS_ERROR_PROPAGATE(_dgs_read(reader, out->shs, count * numShCoeff * 3 * sizeof(uint8_t)));
	}

	if(dynamic)
	{
		DGS_ERROR_PROPAGATE(_dgs_read(reader, out->velocities, count * 4 * sizeof(float)));
	}

	//return:
	//---------------
cleanup:
	return retval;
}

static DGSerror _dgs_read(DGSreader* reader, void* data, uint64_t size)
{
	if(reader->isFile)
	{
		if(fread(data, (size_t)size, 1, reader->file) < 1)
		{
			DGS_LOG_ERROR("failed to read from file");
			return DGS_ERROR_FILE_READ;
		}
	}
	else
	{
		if(reader->buf.pos + size > reader->buf.size)
		{
			DGS_LOG_ERROR("attempting to read past end of buffer");
			return DGS_ERROR_INVALID_INPUT;
		}

		memcpy(data, reader->buf.buf + reader->buf.pos, size);
		reader->buf.pos += size;
	}

	return DGS_SUCCESS;
}