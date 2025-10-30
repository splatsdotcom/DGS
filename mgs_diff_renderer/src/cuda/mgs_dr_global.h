/* mgs_dr_global.h
 *
 * contains utility functions, macros, and global config
 * for the differentiable renderer
 */

#ifndef MGS_DR_GLOBAL_H
#define MGS_DR_GLOBAL_H

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <chrono>

#define QM_FUNC_ATTRIBS __device__ static inline
#include "../external/QuickMath/quickmath.h"

// only enable for personal testing:
// #define MGS_DR_PROFILE

#define MGS_DR_TILE_SIZE 16
#define MGS_DR_TILE_LEN (MGS_DR_TILE_SIZE * MGS_DR_TILE_SIZE)

#define MGS_DR_MAX_ALPHA 0.99f
#define MGS_DR_MIN_ALPHA (1.0f / 255.0f)
#define MGS_DR_ACCUM_ALPHA_CUTOFF 0.00001f

//-------------------------------------------//

struct MGSDRgaussians
{
	uint32_t count;

	float* __restrict__ means;
	float* __restrict__ scales;
	float* __restrict__ rotations;
	float* __restrict__ opacities;
	float* __restrict__ harmonics;
};

struct MGSDRsettings
{
	uint32_t width;
	uint32_t height;

	QMmat4 view;
	QMmat4 proj;
	QMmat4 viewProj;

	float focalX;
	float focalY;

	bool debug;
};

struct MGSDRcov3D
{
	float m00, m01, m02;
	float      m11, m12;
	float           m22;
};

//-------------------------------------------//

__device__ __host__ static __forceinline__ uint32_t _mgs_ceildivide32(uint32_t a, uint32_t b)
{
	return (a + b - 1) / b;
}

//-------------------------------------------//

#define MGS_DR_CUDA_ERROR_CHECK(s)                                        \
	s;                                                                    \
	if(settings.debug)                                                    \
	{                                                                     \
		cudaError error = cudaDeviceSynchronize();                        \
		if(error != cudaSuccess)                                          \
		{                                                                 \
			std::cerr << std::endl << "MGSDR: CUDA error in \"" <<        \
				__FILENAME__ << "\" at line " << __LINE__ <<              \
				": \"" << cudaGetErrorString(error) << "\"" << std::endl; \
			throw std::runtime_error("MGSDR CUDA error");                 \
		}                                                                 \
	}

#ifdef MGS_DR_PROFILE
	#define MGS_DR_PROFILE_REGION_START(name) cudaDeviceSynchronize(); auto tStart##name = std::chrono::high_resolution_clock::now()
	#define MGS_DR_PROFILE_REGION_END(name) cudaDeviceSynchronize(); auto tEnd##name = std::chrono::high_resolution_clock::now()

	#define MGS_DR_PROFILE_REGION_TIME(name) std::chrono::duration_cast<std::chrono::microseconds>(tEnd##name - tStart##name).count() / 1000.0
#else
	#define MGS_DR_PROFILE_REGION_START(name)
	#define MGS_DR_PROFILE_REGION_END(name)

	#define MGS_DR_PROFILE_REGION_TIME(name) 0.0
#endif

#endif //#ifndef MGS_DR_GLOBAL_H
