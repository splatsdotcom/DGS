#define __FILENAME__ "mgs_dr_forward.cu"

#include "mgs_dr_buffers.h"

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include "mgs_dr_global.h"

//-------------------------------------------//

//TODO: should we pass debug here and validate the empty sum/sort calls?

MGSDRrenderBuffers::MGSDRrenderBuffers(uint8_t* mem, uint32_t count) :
	m_mem(mem), m_count(count)
{

}

MGSDRgeomBuffers::MGSDRgeomBuffers(uint8_t* mem, uint32_t count) :
	MGSDRrenderBuffers(mem, count)
{
	pixCenters       = bump<QMvec2>();
	pixRadii         = bump<float>();
	depths           = bump<float>();
	tilesTouched     = bump<uint32_t>();
	covs             = bump<MGSDRcov3D>();
	conicOpacity     = bump<QMvec4>();
	rgb              = bump<QMvec3>();
	tilesTouchedScan = bump<uint32_t>();

	cub::DeviceScan::InclusiveSum(nullptr, tilesTouchedScanTempSize, tilesTouched, tilesTouchedScan, count);
	tilesTouchedScanTemp = bump<uint8_t>(tilesTouchedScanTempSize);
}

MGSDRbinningBuffers::MGSDRbinningBuffers(uint8_t* mem, uint32_t count) :
	MGSDRrenderBuffers(mem, count)
{
	keys          = bump<uint64_t>();
	indices       = bump<uint32_t>();
	keysSorted    = bump<uint64_t>();
	indicesSorted = bump<uint32_t>();

	cub::DeviceRadixSort::SortPairs(nullptr, sortTempSize, keys, keysSorted, indices, indicesSorted, count);
	sortTemp = bump<uint8_t>(sortTempSize);
}

MGSDRimageBuffers::MGSDRimageBuffers(uint8_t* mem, uint32_t count) :
	MGSDRrenderBuffers(mem, count)
{
	//TODO: this is wasteful! only need 1 per tile, can also get away with just a uint32_t
	tileRanges = bump<uint2>();
	accumAlpha = bump<float>();
	numContributors = bump<uint32_t>();
}

MGSDRderivativeBuffers::MGSDRderivativeBuffers(uint8_t* mem, uint32_t count) :
	MGSDRrenderBuffers(mem, count)
{
	dLdPixCenters = bump<QMvec2>();
	dLdConics     = bump<QMvec3>();
	dLdColors     = bump<QMvec3>();
}