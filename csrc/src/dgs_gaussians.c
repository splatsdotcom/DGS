#define __FILENAME__ "dgs_error.c"

#include <math.h>
#include "dgs_gaussians.h"
#include "dgs_log.h"

//-------------------------------------------//

DGSerror dgs_gaussians_allocate(uint32_t count, uint32_t shDegree, dgs_bool_t dynamic, DGSgaussians* out)
{
	DGSerror retval = DGS_SUCCESS;

	DGS_STRUCTURE_CLEAR(out);

	//validate:
	//---------------
	if(count == 0)
	{
		DGS_LOG_ERROR("gaussian count must be positive");
		retval = DGS_ERROR_INVALID_ARGUMENTS;
		goto cleanup;
	}

	if(shDegree > DGS_GAUSSIANS_MAX_SH_DEGREE)
	{
		DGS_LOG_ERROR("gaussian spherical harmomic degree must be less than DGS_GAUSSIANS_MAX_SH_DEGREE");
		retval = DGS_ERROR_INVALID_ARGUMENTS;
		goto cleanup;
	}

	//initialize struct:
	//---------------
	out->count = count;
	out->shDegree = shDegree;
	out->dynamic = dynamic;

	out->scaleMin =  INFINITY;
	out->scaleMax = -INFINITY;
	out->colorMin =  INFINITY;
	out->colorMax = -INFINITY;
	out->shMin =  INFINITY;
	out->shMax = -INFINITY;

	//allocate:
	//---------------
	out->means = (QMvec4*)DGS_MALLOC(count * sizeof(QMvec4));
	DGS_MALLOC_CHECK(out->means);

	out->scales = (uint16_t*)DGS_MALLOC(count * 3 * sizeof(uint16_t));
	DGS_MALLOC_CHECK(out->scales);

	out->rotations = (uint16_t*)DGS_MALLOC(count * 3 * sizeof(uint16_t));
	DGS_MALLOC_CHECK(out->rotations);

	out->opacities = (uint8_t*)DGS_MALLOC(count * sizeof(uint8_t));
	DGS_MALLOC_CHECK(out->opacities);

	out->colors = (uint16_t*)DGS_MALLOC(count * 3 * sizeof(uint16_t));
	DGS_MALLOC_CHECK(out->colors);

	if(shDegree > 0)
	{
		uint32_t numCoeffs = (shDegree + 1) * (shDegree + 1) - 1;

		out->shs = (uint8_t*)DGS_MALLOC(count * numCoeffs * 3 * sizeof(uint8_t));
		DGS_MALLOC_CHECK(out->shs);
	}

	if(dynamic)
	{
		out->velocities = (QMvec4*)DGS_MALLOC(count * sizeof(QMvec4));
		DGS_MALLOC_CHECK(out->velocities);
	}

	//return:
	//---------------
cleanup:
	if(retval != DGS_SUCCESS)
		dgs_gaussians_free(out);

	return retval;
}

void dgs_gaussians_free(DGSgaussians* g)
{
	if(g->means)
		DGS_FREE(g->means);
	if(g->scales)
		DGS_FREE(g->scales);
	if(g->rotations)
		DGS_FREE(g->rotations);
	if(g->opacities)
		DGS_FREE(g->opacities);
	if(g->colors)
		DGS_FREE(g->colors);

	if(g->shs)
		DGS_FREE(g->shs);

	if(g->velocities)
		DGS_FREE(g->velocities);

	DGS_STRUCTURE_CLEAR(g);
}

DGSerror dgs_gaussians_combine(const DGSgaussians* g1, const DGSgaussians* g2, DGSgaussians* out)
{
	DGSerror retval = DGS_SUCCESS;

	DGS_STRUCTURE_CLEAR(out);

	//validate:
	//---------------
	if(g1->shDegree != g2->shDegree) //TODO: make this work
	{
		DGS_LOG_ERROR("cannot combine gaussians with different shDegree");
		return DGS_ERROR_INVALID_INPUT;
	}

	//compute metadata:
	//---------------
	uint32_t count = g1->count + g2->count;
	uint32_t shDegree = g1->shDegree;
	dgs_bool_t dynamic = (g1->dynamic || g2->dynamic);

	float scaleMinOut = DGS_MIN(g1->scaleMin, g2->scaleMin);
	float scaleMaxOut = DGS_MAX(g1->scaleMax, g2->scaleMax);
	float colorMinOut = DGS_MIN(g1->colorMin, g2->colorMin);
	float colorMaxOut = DGS_MAX(g1->colorMax, g2->colorMax);
	float shMinOut    = DGS_MIN(g1->shMin, g2->shMin);
	float shMaxOut    = DGS_MAX(g1->shMax, g2->shMax);

	//allocate output:
	//---------------
	DGS_ERROR_PROPAGATE(
		dgs_gaussians_allocate(count, shDegree, dynamic, out)
	);

	out->scaleMin = scaleMinOut;
	out->scaleMax = scaleMaxOut;
	out->colorMin = colorMinOut;
	out->colorMax = colorMaxOut;
	out->shMin    = shMinOut;
	out->shMax    = shMaxOut;

	//copy non-quantized data:
	//---------------
	memcpy(out->means            , g1->means, sizeof(QMvec4) * g1->count);
	memcpy(out->means + g1->count, g2->means, sizeof(QMvec4) * g2->count);

	memcpy(out->rotations                , g1->rotations, sizeof(uint16_t) * 3 * g1->count);
	memcpy(out->rotations + 3 * g1->count, g2->rotations, sizeof(uint16_t) * 3 * g2->count);

	memcpy(out->opacities            , g1->opacities, sizeof(uint8_t) * g1->count);
	memcpy(out->opacities + g1->count, g2->opacities, sizeof(uint8_t) * g2->count);

	//re-normalize and copy scales:
	//---------------
	float scaleScaleOut = (scaleMaxOut - scaleMinOut);

	for(uint32_t i = 0; i < count; i++)
	for(uint32_t j = 0; j < 3; j++)
	{
		uint16_t v;
		float offset, scale;

		if(i < g1->count)
		{
			v = g1->scales[i * 3 + j];
			offset = g1->scaleMin;
			scale = g1->scaleMax - g1->scaleMin;
		}
		else
		{
			v = g2->scales[(i - g1->count) * 3 + j];
			offset = g2->scaleMin;
			scale = g2->scaleMax - g2->scaleMin;
		}

		float vf = ((float)v / UINT16_MAX) * scale + offset;
		float vn = (vf - scaleMinOut) / scaleScaleOut;
		out->scales[i * 3 + j] = (uint16_t)(vn * UINT16_MAX);
	}

	//re-normalize and copy colors:
	//---------------
	float colorScaleOut = (colorMaxOut - colorMinOut);

	for(uint32_t i = 0; i < count; i++)
	for(uint32_t j = 0; j < 3; j++)
	{
		uint16_t v;
		float offset, scale;

		if(i < g1->count)
		{
			v = g1->colors[i * 3 + j];
			offset = g1->colorMin;
			scale = g1->colorMax - g1->colorMin;
		}
		else
		{
			v = g2->colors[(i - g1->count) * 3 + j];
			offset = g2->colorMin;
			scale = g2->colorMax - g2->colorMin;
		}

		float vf = ((float)v / UINT16_MAX) * scale + offset;
		float vn = (vf - colorMinOut) / colorScaleOut;
		out->colors[i * 3 + j] = (uint16_t)(vn * UINT16_MAX);
	}

	//re-normalize and copy shs:
	//---------------
	float shRangeOut = (shMaxOut - shMinOut);
	uint32_t numShCoeff = (shDegree + 1) * (shDegree + 1) - 1;

	for(uint32_t i = 0; i < count; i++)
	for(uint32_t j = 0; j < numShCoeff * 3; j++)
	{
		uint8_t v;
		float offset, scale;

		if(i < g1->count)
		{
			v = g1->shs[i * numShCoeff * 3 + j];
			offset = g1->shMin;
			scale = g1->shMax - g1->shMin;
		}
		else
		{
			v = g2->shs[(i - g1->count) * numShCoeff * 3 + j];
			offset = g2->shMin;
			scale = g2->shMax - g2->shMin;
		}

		float vf = ((float)v / UINT8_MAX) * scale + offset;
		float vn = (vf - shMinOut) / shRangeOut;
		out->shs[i * numShCoeff * 3 + j] = (uint8_t)(vn * UINT8_MAX);
	}

	//copy velocities:
	//---------------
	if(dynamic)
	{
		if(g1->dynamic)
			memcpy(out->velocities, g1->velocities, g1->count * sizeof(QMvec4));
		else
			memset(out->velocities, 0, g1->count * sizeof(QMvec4));

		if(g2->dynamic)
			memcpy(out->velocities + g1->count, g2->velocities, g2->count * sizeof(QMvec4));
		else
			memset(out->velocities + g1->count, 0, g2->count * sizeof(QMvec4));
	}

	//cleanup + return:
	//---------------
cleanup:
	if(retval != DGS_SUCCESS)
		dgs_gaussians_free(out);
	
	return retval;
}

DGSerror dgs_gaussiansf_allocate(uint32_t count, uint32_t shDegree, dgs_bool_t dynamic, DGSgaussiansF* out)
{
	DGSerror retval = DGS_SUCCESS;

	DGS_STRUCTURE_CLEAR(out);

	//validate:
	//---------------
	if(count == 0)
	{
		DGS_LOG_ERROR("gaussian count must be positive");
		retval = DGS_ERROR_INVALID_ARGUMENTS;
		goto cleanup;
	}

	if(shDegree > DGS_GAUSSIANS_MAX_SH_DEGREE)
	{
		DGS_LOG_ERROR("gaussian spherical harmomic degree must be less than DGS_GAUSSIANS_MAX_SH_DEGREE");
		retval = DGS_ERROR_INVALID_ARGUMENTS;
		goto cleanup;
	}

	//initialize struct:
	//---------------
	out->count = count;
	out->shDegree = shDegree;
	out->dynamic = dynamic;

	//allocate:
	//---------------
	out->means = (QMvec3*)DGS_MALLOC(count * sizeof(QMvec3));
	DGS_MALLOC_CHECK(out->means);

	out->scales = (QMvec3*)DGS_MALLOC(count * sizeof(QMvec3));
	DGS_MALLOC_CHECK(out->scales);

	out->rotations = (QMquaternion*)DGS_MALLOC(count * sizeof(QMquaternion));
	DGS_MALLOC_CHECK(out->rotations);

	out->opacities = (float*)DGS_MALLOC(count * sizeof(float));
	DGS_MALLOC_CHECK(out->opacities);

	uint32_t numCoeffs = (shDegree + 1) * (shDegree + 1);
	out->shs = (float*)DGS_MALLOC(count * numCoeffs * 3 * sizeof(float));
	DGS_MALLOC_CHECK(out->shs);

	if(dynamic)
	{
		out->velocities = (QMvec3*)DGS_MALLOC(count * sizeof(QMvec3));
		DGS_MALLOC_CHECK(out->velocities);

		out->tMeans = (float*)DGS_MALLOC(count * sizeof(float));
		DGS_MALLOC_CHECK(out->tMeans);

		out->tStdevs = (float*)DGS_MALLOC(count * sizeof(float));
		DGS_MALLOC_CHECK(out->tStdevs);
	}

	//return:
	//---------------
cleanup:
	if(retval != DGS_SUCCESS)
		dgs_gaussiansf_free(out);

	return retval;
}

void dgs_gaussiansf_free(DGSgaussiansF* g)
{
	if(g->means)
		DGS_FREE(g->means);
	if(g->scales)
		DGS_FREE(g->scales);
	if(g->rotations)
		DGS_FREE(g->rotations);
	if(g->opacities)
		DGS_FREE(g->opacities);
	if(g->shs)
		DGS_FREE(g->shs);

	if(g->velocities)
		DGS_FREE(g->velocities);
	if(g->tMeans)
		DGS_FREE(g->tMeans);
	if(g->tStdevs)
		DGS_FREE(g->tStdevs);

	DGS_STRUCTURE_CLEAR(g);
}

DGSerror dgs_gaussians_to_fp32(const DGSgaussians* src, DGSgaussiansF* dst)
{
	DGSerror retval = DGS_SUCCESS;

	DGS_STRUCTURE_CLEAR(dst);

	//allocate:
	//---------------
	DGS_ERROR_PROPAGATE(dgs_gaussiansf_allocate(
		src->count, src->shDegree, src->dynamic, dst
	));
	
	//convert:
	//---------------

	//TODO

	//return:
	//---------------
cleanup:
	if(retval != DGS_SUCCESS)
		dgs_gaussiansf_free(dst);

	return retval;
}

DGSerror dgs_gaussians_from_fp32(const DGSgaussiansF* src, DGSgaussians* dst)
{
	DGSerror retval = DGS_SUCCESS;

	DGS_STRUCTURE_CLEAR(dst);

	//allocate:
	//---------------
	DGS_ERROR_PROPAGATE(dgs_gaussians_allocate(
		src->count, src->shDegree, src->dynamic, dst
	));
	
	//compute normalization constants:
	//---------------
	uint32_t numShCoeffs = (src->shDegree + 1) * (src->shDegree + 1);

	dst->scaleMin =  INFINITY;
	dst->scaleMax = -INFINITY;
	dst->colorMin =  INFINITY;
	dst->colorMax = -INFINITY;
	dst->shMin =  INFINITY;
	dst->shMax = -INFINITY;

	for(uint32_t i = 0; i < src->count; i++)
	{
		uint32_t idx = i * (numShCoeffs * 3);

		for(uint32_t j = 0; j < 3; j++)
		{
			dst->scaleMin = DGS_MIN(dst->scaleMin, src->scales[idx].v[j]);
			dst->scaleMax = DGS_MAX(dst->scaleMax, src->scales[idx].v[j]);
		}

		for(uint32_t j = 0; j < 3; j++)
		{
			dst->colorMin = DGS_MIN(dst->colorMin, src->shs[idx + j]);
			dst->colorMax = DGS_MAX(dst->colorMax, src->shs[idx + j]);
		}

		for(uint32_t j = 3; j < numShCoeffs * 3; j++)
		{
			dst->shMin = DGS_MIN(dst->shMin, src->shs[idx + j]);
			dst->shMax = DGS_MAX(dst->shMax, src->shs[idx + j]);
		}
	}
		
	//convert:
	//---------------
	float scaleScale = 1.0f / (dst->scaleMax - dst->scaleMin);
	float colorScale = 1.0f / (dst->colorMax - dst->colorMin);
	float shScale = 1.0f / (dst->shMax - dst->shMin);

	for(uint32_t i = 0; i < src->count; i++)
	{
		uint32_t idxShSrc = i * (numShCoeffs * 3);
		uint32_t idxShDst = i * (numShCoeffs * 3 - 3);

		//mean
		dst->means[i] = (QMvec4){
			src->means[i].x,
			src->means[i].y,
			src->means[i].z,
			src->dynamic ? src->tMeans[i] : 0.5f
		};

		//scale
		for(uint32_t j = 0; j < 3; j++)
			dst->scales[i * 3 + j] = (uint16_t)((src->scales[i].v[j] - dst->scaleMin) * scaleScale * UINT16_MAX);

		//rotation
		QMquaternion norm = qm_quaternion_normalize(src->rotations[i]);
		
		for(uint32_t j = 0; j < 3; j++)
			dst->rotations[i * 3 + j] = (uint16_t)(norm.q[j] * UINT16_MAX);

		//opacity
		dst->opacities[i] = (uint8_t)(src->opacities[i] * UINT8_MAX);

		//color
		for(uint32_t j = 0; j < 3; j++)
			dst->colors[i * 3 + j] = (uint16_t)((src->shs[idxShSrc + j] - dst->colorMin) * colorScale * UINT16_MAX);

		//sh
		for(uint32_t j = 0; j < numShCoeffs * 3 - 3; j++)
			dst->shs[idxShDst + j] = (uint8_t)((src->shs[idxShSrc + j + 3] - dst->shMin) * shScale * UINT8_MAX);

		//velocity
		if(src->dynamic)
		{
			dst->velocities[i] = (QMvec4){
				src->velocities[i].x,
				src->velocities[i].y,
				src->velocities[i].z,
				src->tStdevs[i]
			};
		}
	}

	//return:
	//---------------
cleanup:
	if(retval != DGS_SUCCESS)
		dgs_gaussians_free(dst);

	return retval;
}
