/* mgs_dr_buffers.h
 *
 * contains declarations for the temp buffers 
 * used in the differentiable rendering process
 */

#ifndef MGS_DR_BUFFERS_H
#define MGS_DR_BUFFERS_H

#include <math.h>
#include <stdint.h>
#include "mgs_dr_global.h"

#define QM_FUNC_ATTRIBS __device__ static inline
#include "../external/QuickMath/quickmath.h"

//-------------------------------------------//

#define MGS_DR_L1_CACHE_ALIGNMENT 128

//-------------------------------------------//

class MGSDRrenderBuffers
{
public:
	MGSDRrenderBuffers(uint8_t* mem, uint32_t count);

	template<typename T>
	static uint64_t required_mem(uint32_t count)
	{
		T buffer(nullptr, count);
		return reinterpret_cast<uint64_t>(buffer.m_mem);
	}
	
private:
	uint8_t* m_mem;
	uint32_t m_count;

protected:
	template<typename T>
	T* bump()
	{
		uint64_t alignment = std::max((uint64_t)MGS_DR_L1_CACHE_ALIGNMENT, (uint64_t)sizeof(T));
		uint64_t offset = (reinterpret_cast<uintptr_t>(m_mem) + alignment - 1) & ~(alignment - 1);
		
		T* ptr = reinterpret_cast<T*>(offset);
		m_mem = reinterpret_cast<uint8_t*>(ptr + m_count);

		return ptr;
	}

	template<typename T>
	T* bump(uint64_t size)
	{
		uint64_t alignment = std::max((uint64_t)MGS_DR_L1_CACHE_ALIGNMENT, (uint64_t)sizeof(T));
		uint64_t offset = (reinterpret_cast<uintptr_t>(m_mem) + alignment - 1) & ~(alignment - 1);
		
		T* ptr = reinterpret_cast<T*>(offset);
		m_mem = reinterpret_cast<uint8_t*>(ptr) + size;

		return ptr;
	}
};

struct MGSDRgeomBuffers : public MGSDRrenderBuffers
{
public:
	MGSDRgeomBuffers(uint8_t* mem, uint32_t count);

	QMvec2*     __restrict__ pixCenters;
	float*      __restrict__ pixRadii;
	float*      __restrict__ depths;
	uint32_t*   __restrict__ tilesTouched;
	uint32_t*   __restrict__ tilesTouchedScan;
	MGSDRcov3D* __restrict__ covs;
	QMvec4*     __restrict__ conicOpacity;
	QMvec3*     __restrict__ rgb;

	size_t tilesTouchedScanTempSize;
	uint8_t* __restrict__ tilesTouchedScanTemp;
};

struct MGSDRbinningBuffers : public MGSDRrenderBuffers
{
public:
	MGSDRbinningBuffers(uint8_t* mem, uint32_t count);

	uint64_t* __restrict__ keys;
	uint32_t* __restrict__ indices;
	uint64_t* __restrict__ keysSorted;
	uint32_t* __restrict__ indicesSorted;

	size_t sortTempSize;
	uint8_t* sortTemp;
};

struct MGSDRimageBuffers : public MGSDRrenderBuffers
{
public:
	MGSDRimageBuffers(uint8_t* mem, uint32_t count);

	uint2*    __restrict__ tileRanges;
	float*    __restrict__ accumAlpha;
	uint32_t* __restrict__ numContributors;
};

struct MGSDRderivativeBuffers : public MGSDRrenderBuffers
{
public:
	MGSDRderivativeBuffers(uint8_t* mem, uint32_t count);

	QMvec2* __restrict__ dLdPixCenters;
	QMvec3* __restrict__ dLdConics;
	QMvec3* __restrict__ dLdColors;
};

#endif //#ifndef MGS_DR_BUFFERS_H