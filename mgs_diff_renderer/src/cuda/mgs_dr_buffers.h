/* mgs_dr_buffers.h
 *
 * contains declarations for the temp buffers 
 * used in the differentiable rendering process
 */

#ifndef MGS_DR_BUFFERS_H
#define MGS_DR_BUFFERS_H

#include <math.h>
#include <stdint.h>

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
};

struct MGSDRgeomBuffers : public MGSDRrenderBuffers
{
public:
	MGSDRgeomBuffers(uint8_t* mem, uint32_t count);

	uint2* pixCenters;
	float* pixRadii;
	float* depths;
	uint32_t* tilesTouched;
	float4* conic;
	float3* rgb;
};

struct MGSDRbinningBuffers : public MGSDRrenderBuffers
{
public:
	MGSDRbinningBuffers(uint8_t* mem, uint32_t count);

	uint32_t* test; //todo 
};

#endif //#ifndef MGS_DR_BUFFERS_H