/* mgs_gaussians.h
 *
 * contains the definitions of gaussians structs
 */

#include "mgs_global.h"
#include "mgs_error.h"
#include "QuickMath/quickmath.h"

#ifndef MGS_GAUSSIANS_H
#define MGS_GAUSSIANS_H

#define MGS_GAUSSIANS_MAX_SH_DEGREE 3

//-------------------------------------------//

/**
 * a group of dynamic gaussians
 */
typedef struct MGSgaussians
{
	//TODO: find optimal memory layout
	//TODO: store precomputed covariance? or rot + scale? probably want to match MGSgaussiansF

	uint32_t count;
	uint32_t shDegree;
	mgs_bool_t dynamic;

	float colorMin;
	float colorMax;
	float shMin;
	float shMax;

	QMvec4* means;      // (mean x, mean y, mean z, mean t) fp32
	float* covariances; // (m00, m01, m02, m11, m12, m22) fp32
	uint8_t* opacities; // (a) unorm8 in [0.0, 1.0]
	uint16_t* colors;   // (r, g, b) unorm16 in [colorMin, colorMax]
	uint8_t* shs;       // (shDegree + 1)^2 - 1 (r, g, b) unorm8 in [shMin, shMax], NULL if shDegree == 0

	QMvec4* velocities; // (vel x, vel y, vel z, t-stdev) fp32, NULL if dynamic == MGS_FALSE
} MGSgaussians;

/**
 * a group of dynamic gaussians, stored in full fp32 precision
 */
typedef struct MGSgaussiansF
{
	uint32_t count;
	uint32_t shDegree;
	mgs_bool_t dynamic;

	QMvec3* means;
	QMvec3* scales;
	QMquaternion* rotations;
	float* opacities;
	float* shs;         // (shDegree + 1)^2 (r, g, b)

	QMvec3* velocities; // (vel x, vel y, vel z, t-stdev), NULL if dynamic == MGS_FALSE
	float* tMeans;      // NULL if dynamic == MGS_FALSE
	float* tStdevs;     // NULL if dynamic == MGS_FALSE
} MGSgaussiansF;

//-------------------------------------------//

/**
 * allocates memory for MGSgaussians, call mgs_gaussians_free to free
 */
MGS_API MGSerror mgs_gaussians_allocate(uint32_t count, uint32_t shDegree, mgs_bool_t dynamic, MGSgaussians* out);

/**
 * frees memory allocated from mgs_gaussians_allocate
 */
MGS_API void mgs_gaussians_free(MGSgaussians* gaussians);

/**
 * combines 2 sets of gaussians, if at least 1 of them is dynamic, the combined gaussians will also be dynamic
 */
MGS_API MGSerror mgs_gaussians_combine(const MGSgaussians* g1, const MGSgaussians* g2, MGSgaussians* out);


/**
 * allocates memory for MGSgaussiansF, call mgs_gaussians_free to free
 */
MGS_API MGSerror mgs_gaussiansf_allocate(uint32_t count, uint32_t shDegree, mgs_bool_t dynamic, MGSgaussiansF* out);

/**
 * frees memory allocated from mgs_gaussians_allocate
 */
MGS_API void mgs_gaussiansf_free(MGSgaussiansF* gaussians);

/**
 * converts MGSgaussians to MGSgaussiansF
 */
MGS_API MGSerror mgs_gaussians_to_fp32(const MGSgaussians* src, MGSgaussiansF* dst);

/**
 * converts MGSgaussiansF to MGSgaussians, note that this is lossy due to quantization
 */
MGS_API MGSerror mgs_gaussians_from_fp32(const MGSgaussiansF* src, MGSgaussians* dst);

#endif //#ifndef MGS_GAUSSIANS_H
