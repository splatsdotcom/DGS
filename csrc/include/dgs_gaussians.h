/* dgs_gaussians.h
 *
 * contains the definitions of gaussians structs
 */

#include "dgs_global.h"
#include "dgs_error.h"
#include "QuickMath/quickmath.h"

#ifndef DGS_GAUSSIANS_H
#define DGS_GAUSSIANS_H

#define DGS_GAUSSIANS_MAX_SH_DEGREE 3

//-------------------------------------------//

/**
 * a group of dynamic gaussians
 */
typedef struct DGSgaussians
{
	//TODO: find optimal memory layout
	//TODO: store precomputed covariance? or rot + scale? probably want to match DGSgaussiansF

	uint32_t count;
	uint32_t shDegree;
	dgs_bool_t dynamic;

	float colorMin;
	float colorMax;
	float shMin;
	float shMax;

	QMvec4* means;      // (mean x, mean y, mean z, mean t) fp32
	float* covariances; // (m00, m01, m02, m11, m12, m22) fp32
	uint8_t* opacities; // (a) unorm8 in [0.0, 1.0]
	uint16_t* colors;   // (r, g, b) unorm16 in [colorMin, colorMax]
	uint8_t* shs;       // (shDegree + 1)^2 - 1 (r, g, b) unorm8 in [shMin, shMax], NULL if shDegree == 0

	QMvec4* velocities; // (vel x, vel y, vel z, t-stdev) fp32, NULL if dynamic == DGS_FALSE
} DGSgaussians;

/**
 * a group of dynamic gaussians, stored in full fp32 precision
 */
typedef struct DGSgaussiansF
{
	uint32_t count;
	uint32_t shDegree;
	dgs_bool_t dynamic;

	QMvec3* means;
	QMvec3* scales;
	QMquaternion* rotations;
	float* opacities;
	float* shs;         // (shDegree + 1)^2 (r, g, b)

	QMvec3* velocities; // (vel x, vel y, vel z, t-stdev), NULL if dynamic == DGS_FALSE
	float* tMeans;      // NULL if dynamic == DGS_FALSE
	float* tStdevs;     // NULL if dynamic == DGS_FALSE
} DGSgaussiansF;

//-------------------------------------------//

/**
 * allocates memory for DGSgaussians, call dgs_gaussians_free to free
 */
DGS_API DGSerror dgs_gaussians_allocate(uint32_t count, uint32_t shDegree, dgs_bool_t dynamic, DGSgaussians* out);

/**
 * frees memory allocated from dgs_gaussians_allocate
 */
DGS_API void dgs_gaussians_free(DGSgaussians* gaussians);

/**
 * combines 2 sets of gaussians, if at least 1 of them is dynamic, the combined gaussians will also be dynamic
 */
DGS_API DGSerror dgs_gaussians_combine(const DGSgaussians* g1, const DGSgaussians* g2, DGSgaussians* out);


/**
 * allocates memory for DGSgaussiansF, call dgs_gaussians_free to free
 */
DGS_API DGSerror dgs_gaussiansf_allocate(uint32_t count, uint32_t shDegree, dgs_bool_t dynamic, DGSgaussiansF* out);

/**
 * frees memory allocated from dgs_gaussians_allocate
 */
DGS_API void dgs_gaussiansf_free(DGSgaussiansF* gaussians);

/**
 * converts DGSgaussians to DGSgaussiansF
 */
DGS_API DGSerror dgs_gaussians_to_fp32(const DGSgaussians* src, DGSgaussiansF* dst);

/**
 * converts DGSgaussiansF to DGSgaussians, note that this is lossy due to quantization
 */
DGS_API DGSerror dgs_gaussians_from_fp32(const DGSgaussiansF* src, DGSgaussians* dst);

#endif //#ifndef DGS_GAUSSIANS_H
