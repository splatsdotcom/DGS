#define __FILENAME__ "mgs_error.c"

#include <math.h>
#include "mgs_gaussians.h"
#include "mgs_log.h"

//-------------------------------------------//

MGSerror mgs_gaussians_allocate(uint32_t count, uint32_t shDegree, mgs_bool_t dynamic, MGSgaussians* out)
{
	MGSerror retval = MGS_SUCCESS;

	MGS_STRUCTURE_CLEAR(out);

	//validate:
	//---------------
	if(count == 0)
	{
		MGS_LOG_ERROR("gaussian count must be positive");
		retval = MGS_ERROR_INVALID_ARGUMENTS;
		goto cleanup;
	}

	if(shDegree > MGS_GAUSSIANS_MAX_SH_DEGREE)
	{
		MGS_LOG_ERROR("gaussian spherical harmomic degree must be less than MGS_GAUSSIANS_MAX_SH_DEGREE");
		retval = MGS_ERROR_INVALID_ARGUMENTS;
		goto cleanup;
	}

	//initialize struct:
	//---------------
	out->count = count;
	out->shDegree = shDegree;
	out->dynamic = dynamic;

	out->colorMin = -1.0f;
	out->colorMax =  1.0f;
	out->shMin = -1.0f;
	out->shMax =  1.0f;

	//allocate:
	//---------------
	out->means = (QMvec4*)MGS_MALLOC(count * sizeof(QMvec4));
	MGS_MALLOC_CHECK(out->means);

	out->covariances = (float*)MGS_MALLOC(count * 6 * sizeof(float));
	MGS_MALLOC_CHECK(out->covariances);

	out->opacities = (uint8_t*)MGS_MALLOC(count * sizeof(uint8_t));
	MGS_MALLOC_CHECK(out->opacities);

	out->colors = (uint16_t*)MGS_MALLOC(count * 3 * sizeof(uint16_t));
	MGS_MALLOC_CHECK(out->colors);

	if(shDegree > 0)
	{
		uint32_t numCoeffs = (shDegree + 1) * (shDegree + 1) - 1;

		out->shs = (uint8_t*)MGS_MALLOC(count * numCoeffs * 3 * sizeof(uint8_t));
		MGS_MALLOC_CHECK(out->shs);
	}

	if(dynamic)
	{
		out->velocities = (QMvec4*)MGS_MALLOC(count * sizeof(QMvec4));
		MGS_MALLOC_CHECK(out->velocities);
	}

	//return:
	//---------------
cleanup:
	if(retval != MGS_SUCCESS)
		mgs_gaussians_free(out);

	return retval;
}

void mgs_gaussians_free(MGSgaussians* g)
{
	if(g->means)
		MGS_FREE(g->means);
	if(g->covariances)
		MGS_FREE(g->covariances);
	if(g->opacities)
		MGS_FREE(g->opacities);
	if(g->colors)
		MGS_FREE(g->colors);

	if(g->shs)
		MGS_FREE(g->shs);

	if(g->velocities)
		MGS_FREE(g->velocities);

	MGS_STRUCTURE_CLEAR(g);
}

MGSerror mgs_gaussiansf_allocate(uint32_t count, uint32_t shDegree, mgs_bool_t dynamic, MGSgaussiansF* out)
{
	MGSerror retval = MGS_SUCCESS;

	MGS_STRUCTURE_CLEAR(out);

	//validate:
	//---------------
	if(count == 0)
	{
		MGS_LOG_ERROR("gaussian count must be positive");
		retval = MGS_ERROR_INVALID_ARGUMENTS;
		goto cleanup;
	}

	if(shDegree > MGS_GAUSSIANS_MAX_SH_DEGREE)
	{
		MGS_LOG_ERROR("gaussian spherical harmomic degree must be less than MGS_GAUSSIANS_MAX_SH_DEGREE");
		retval = MGS_ERROR_INVALID_ARGUMENTS;
		goto cleanup;
	}

	//initialize struct:
	//---------------
	out->count = count;
	out->shDegree = shDegree;
	out->dynamic = dynamic;

	//allocate:
	//---------------
	out->means = (QMvec3*)MGS_MALLOC(count * sizeof(QMvec3));
	MGS_MALLOC_CHECK(out->means);

	out->scales = (QMvec3*)MGS_MALLOC(count * sizeof(QMvec3));
	MGS_MALLOC_CHECK(out->scales);

	out->rotations = (QMquaternion*)MGS_MALLOC(count * sizeof(QMquaternion));
	MGS_MALLOC_CHECK(out->rotations);

	out->opacities = (float*)MGS_MALLOC(count * sizeof(float));
	MGS_MALLOC_CHECK(out->opacities);

	uint32_t numCoeffs = (shDegree + 1) * (shDegree + 1);
	out->shs = (float*)MGS_MALLOC(count * numCoeffs * 3 * sizeof(float));
	MGS_MALLOC_CHECK(out->shs);

	if(dynamic)
	{
		out->velocities = (QMvec3*)MGS_MALLOC(count * sizeof(QMvec3));
		MGS_MALLOC_CHECK(out->velocities);

		out->tMeans = (float*)MGS_MALLOC(count * sizeof(float));
		MGS_MALLOC_CHECK(out->tMeans);

		out->tStdevs = (float*)MGS_MALLOC(count * sizeof(float));
		MGS_MALLOC_CHECK(out->tStdevs);
	}

	//return:
	//---------------
cleanup:
	if(retval != MGS_SUCCESS)
		mgs_gaussiansf_free(out);

	return retval;
}

void mgs_gaussiansf_free(MGSgaussiansF* g)
{
	if(g->means)
		MGS_FREE(g->means);
	if(g->scales)
		MGS_FREE(g->scales);
	if(g->rotations)
		MGS_FREE(g->rotations);
	if(g->opacities)
		MGS_FREE(g->opacities);
	if(g->shs)
		MGS_FREE(g->shs);

	if(g->velocities)
		MGS_FREE(g->velocities);
	if(g->tMeans)
		MGS_FREE(g->tMeans);
	if(g->tStdevs)
		MGS_FREE(g->tStdevs);

	MGS_STRUCTURE_CLEAR(g);
}

MGSerror mgs_gaussians_to_fp32(const MGSgaussians* src, MGSgaussiansF* dst)
{
	MGSerror retval = MGS_SUCCESS;

	MGS_STRUCTURE_CLEAR(dst);

	//allocate:
	//---------------
	MGS_ERROR_PROPAGATE(mgs_gaussiansf_allocate(
		src->count, src->shDegree, src->dynamic, dst
	));
	
	//convert:
	//---------------

	//TODO

	//return:
	//---------------
cleanup:
	if(retval != MGS_SUCCESS)
		mgs_gaussiansf_free(dst);

	return retval;
}

MGSerror mgs_gaussians_from_fp32(const MGSgaussiansF* src, MGSgaussians* dst)
{
	MGSerror retval = MGS_SUCCESS;

	MGS_STRUCTURE_CLEAR(dst);

	//allocate:
	//---------------
	MGS_ERROR_PROPAGATE(mgs_gaussians_allocate(
		src->count, src->shDegree, src->dynamic, dst
	));
	
	//compute normalization constants:
	//---------------
	uint32_t numShCoeffs = (src->shDegree + 1) * (src->shDegree + 1);

	dst->colorMin =  INFINITY;
	dst->colorMax = -INFINITY;
	dst->shMin =  INFINITY;
	dst->shMax = -INFINITY;

	for(uint32_t i = 0; i < src->count; i++)
	{
		uint32_t idx = i * (numShCoeffs * 3);

		for(uint32_t j = 0; j < 3; j++)
		{
			dst->colorMin = MGS_MIN(dst->colorMin, src->shs[idx + j]);
			dst->colorMax = MGS_MAX(dst->colorMax, src->shs[idx + j]);
		}

		for(uint32_t j = 3; j < numShCoeffs * 3; j++)
		{
			dst->shMin = MGS_MIN(dst->shMin, src->shs[idx + j]);
			dst->shMax = MGS_MAX(dst->shMax, src->shs[idx + j]);
		}
	}
		
	//convert:
	//---------------
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

		//covariance
		QMmat4 M = qm_mat4_mult(
			qm_mat4_scale(src->scales[i]), 
			qm_quaternion_to_mat4(src->rotations[i])
		);
		float covariance[6] = {
			M.m[0][0] * M.m[0][0] + M.m[0][1] * M.m[0][1] + M.m[0][2] * M.m[0][2],
			M.m[0][0] * M.m[1][0] + M.m[0][1] * M.m[1][1] + M.m[0][2] * M.m[1][2],
			M.m[0][0] * M.m[2][0] + M.m[0][1] * M.m[2][1] + M.m[0][2] * M.m[2][2],
			M.m[1][0] * M.m[1][0] + M.m[1][1] * M.m[1][1] + M.m[1][2] * M.m[1][2],
			M.m[1][0] * M.m[2][0] + M.m[1][1] * M.m[2][1] + M.m[1][2] * M.m[2][2],
			M.m[2][0] * M.m[2][0] + M.m[2][1] * M.m[2][1] + M.m[2][2] * M.m[2][2]
		};

		for(uint32_t j = 0; j < 6; j++)
			dst->covariances[i * 6 + j] = 4.0f * covariance[j];

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
	if(retval != MGS_SUCCESS)
		mgs_gaussians_free(dst);

	return retval;
}
