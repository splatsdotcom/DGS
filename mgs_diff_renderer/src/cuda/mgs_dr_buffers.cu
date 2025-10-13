#include "mgs_dr_buffers.h"

//-------------------------------------------//

MGSDRrenderBuffers::MGSDRrenderBuffers(uint8_t* mem, uint32_t count) :
	m_mem(mem), m_count(count)
{

}

MGSDRgeomBuffers::MGSDRgeomBuffers(uint8_t* mem, uint32_t count) :
	MGSDRrenderBuffers(mem, count)
{
	pixCenters = bump<uint2>();
	pixRadii = bump<float>();
	depths = bump<float>();
	tilesTouched = bump<uint32_t>();
	conic = bump<float4>();
	rgb = bump<float3>();
}

MGSDRbinningBuffers::MGSDRbinningBuffers(uint8_t* mem, uint32_t count) :
	MGSDRrenderBuffers(mem, count)
{
	test = bump<uint32_t>();
}