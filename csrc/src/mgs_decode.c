#define __FILENAME__ "mgs_decode.c"

#include "mgs_decode.h"
#include "mgs_log.h"
#include <stdio.h>

//-------------------------------------------//

typedef struct MGSreader
{
	mgs_bool_t isFile;
	
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
} MGSreader;

//-------------------------------------------//

static MGSerror _mgs_decode(MGSreader* reader, MGSgaussians* out, MGSmetadata* outMetadata);

static MGSerror _mgs_read(MGSreader* reader, void* data, uint64_t size);

//-------------------------------------------//

MGSerror mgs_decode_from_file(const char* path, MGSgaussians* out, MGSmetadata* outMetadata)
{
	MGSerror retval = MGS_SUCCESS;

	FILE* in = NULL;
	MGS_STRUCTURE_CLEAR(out);

	//open file:
	//---------------
	in = fopen(path, "rb");
	if(!in)
	{
		MGS_LOG_ERROR("failed to open input file to read");
		retval = MGS_ERROR_FILE_OPEN;
		goto cleanup;
	}	

	//decode:
	//---------------
	MGSreader reader;
	reader.isFile = MGS_TRUE;
	reader.file = in;

	MGS_ERROR_PROPAGATE(_mgs_decode(&reader, out, outMetadata));

	//cleanup + return:
	//---------------
cleanup:
	if(retval != MGS_SUCCESS)
		mgs_gaussians_free(out);
	if(in)
		fclose(in);

	return retval;
}

MGSerror mgs_decode_from_buffer(uint64_t size, const uint8_t* buf, MGSgaussians* out, MGSmetadata* outMetadata)
{
	MGSerror retval = MGS_SUCCESS;

	MGS_STRUCTURE_CLEAR(out);

	//decode:
	//---------------
	MGSreader reader;
	reader.isFile = MGS_FALSE;
	reader.buf.size = size;
	reader.buf.pos = 0;
	reader.buf.buf = buf;

	MGS_ERROR_PROPAGATE(_mgs_decode(&reader, out, outMetadata));

	//cleanup + return:
	//---------------
cleanup:
	if(retval != MGS_SUCCESS)
		mgs_gaussians_free(out);

	return retval;
}

static MGSerror _mgs_decode(MGSreader* reader, MGSgaussians* out, MGSmetadata* outMetadata)
{
	MGSerror retval = MGS_SUCCESS;

	//file header + metadata:
	//---------------
	MGSfileHeader header;

	MGS_ERROR_PROPAGATE(_mgs_read(reader, &header, sizeof(MGSfileHeader)));
	MGS_ERROR_PROPAGATE(_mgs_read(reader, outMetadata, sizeof(MGSmetadata)));

	//validate file header + metadata:
	//---------------
	if(header.magicWord != MGS_MAGIC_WORD)
	{
		MGS_LOG_ERROR("mismatched magic word");
		retval = MGS_ERROR_INVALID_INPUT;
		goto cleanup;
	}

	if(header.version != MGS_VERSION) //TODO: keep versions compatible!
	{
		MGS_LOG_ERROR("mismatched version");
		retval = MGS_ERROR_INVALID_INPUT;
		goto cleanup;
	}

	if(outMetadata->duration < 0.0f)
		MGS_LOG_WARNING("negative duration encountered in metadata");

	//read gaussian properties:
	//---------------
	uint32_t count;
	mgs_bool_t dynamic;
	uint32_t shDegree;

	float colorMin, colorMax;
	float shMin, shMax;

	MGS_ERROR_PROPAGATE(_mgs_read(reader, &count   , sizeof(uint32_t)));
	MGS_ERROR_PROPAGATE(_mgs_read(reader, &dynamic , sizeof(mgs_bool_t)));

	MGS_ERROR_PROPAGATE(_mgs_read(reader, &shDegree, sizeof(uint32_t)));
	
	MGS_ERROR_PROPAGATE(_mgs_read(reader, &colorMin, sizeof(float)));
	MGS_ERROR_PROPAGATE(_mgs_read(reader, &colorMax, sizeof(float)));
	MGS_ERROR_PROPAGATE(_mgs_read(reader, &shMin   , sizeof(float)));
	MGS_ERROR_PROPAGATE(_mgs_read(reader, &shMax   , sizeof(float)));

	//validate gaussian properties:
	//---------------
	if(count == 0)
	{
		MGS_LOG_ERROR("file contains 0 gaussians");
		retval = MGS_ERROR_INVALID_INPUT;
		goto cleanup;
	}

	if(shDegree > MGS_GAUSSIANS_MAX_SH_DEGREE)
	{
		MGS_LOG_ERROR("out of bounds sh degree");
		retval = MGS_ERROR_INVALID_INPUT;
		goto cleanup;
	}

	if(colorMin > colorMax)
	{
		MGS_LOG_ERROR("invalid color normalization coefficients");
		retval = MGS_ERROR_INVALID_INPUT;
		goto cleanup;
	}

	if(shDegree > 0 && shMin > shMax)
	{
		MGS_LOG_ERROR("invalid sh normalization coefficients");
		retval = MGS_ERROR_INVALID_INPUT;
		goto cleanup;
	}
	
	//allocate gaussians:
	//---------------
	MGS_ERROR_PROPAGATE(mgs_gaussians_allocate(
		count, shDegree, dynamic, out
	));

	out->colorMin = colorMin;
	out->colorMax = colorMax;
	out->shMin = shMin;
	out->shMax = shMax;

	//read gaussian data:
	//---------------
	uint32_t numShCoeff = (shDegree + 1) * (shDegree + 1) - 1;

	MGS_ERROR_PROPAGATE(_mgs_read(reader, out->means, count * 4 * sizeof(float)));
	MGS_ERROR_PROPAGATE(_mgs_read(reader, out->covariances, count * 6 * sizeof(float)));
	MGS_ERROR_PROPAGATE(_mgs_read(reader, out->opacities, count * sizeof(uint8_t)));
	MGS_ERROR_PROPAGATE(_mgs_read(reader, out->colors, count * 3 * sizeof(uint16_t)));

	if(numShCoeff > 0)
	{
		MGS_ERROR_PROPAGATE(_mgs_read(reader, out->shs, count * numShCoeff * 3 * sizeof(uint8_t)));
	}

	if(dynamic)
	{
		MGS_ERROR_PROPAGATE(_mgs_read(reader, out->velocities, count * 4 * sizeof(float)));
	}

	//return:
	//---------------
cleanup:
	return retval;
}

static MGSerror _mgs_read(MGSreader* reader, void* data, uint64_t size)
{
	if(reader->isFile)
	{
		if(fread(data, (size_t)size, 1, reader->file) < 1)
		{
			MGS_LOG_ERROR("failed to read from file");
			return MGS_ERROR_FILE_READ;
		}
	}
	else
	{
		if(reader->buf.pos + size > reader->buf.size)
		{
			MGS_LOG_ERROR("attempting to read past end of buffer");
			return MGS_ERROR_INVALID_INPUT;
		}

		memcpy(data, reader->buf.buf + reader->buf.pos, size);
		reader->buf.pos += size;
	}

	return MGS_SUCCESS;
}