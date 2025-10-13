#include "mgs_dr_forward.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <cooperative_groups.h>

#include "mgs_dr_buffers.h"

namespace cg = cooperative_groups;

//-------------------------------------------//

#define MGS_DR_TILE_SIZE 16
#define MGS_DR_PREPROCESS_WORKGROUP_SIZE 64

//-------------------------------------------//

__global__ void __launch_bounds__(MGS_DR_PREPROCESS_WORKGROUP_SIZE)
_mgs_dr_foward_preprocess_kernel(const float* view, const float* proj, float focalX, float focalY, 
                                 uint32_t numGaussians, const float* means, const float* scales, const float* rotations, const float* opacities, const float* colors, const float* harmonics,
                                 uint2* outPixCenters, float* outPixRadii, float* outDepths, uint32_t* outTilesTouched, float4* outConic, float3* outRGB);

__global__ void __launch_bounds__(MGS_DR_TILE_SIZE * MGS_DR_TILE_SIZE) 
_mgs_dr_forward_splat_kernel(uint32_t width, uint32_t height, float* img);

//-------------------------------------------//

uint32_t mgs_dr_forward_cuda(uint32_t outWidth, uint32_t outHeight, float* outImg, const float* view, const float* proj, float focalX, float focalY,
                             uint32_t numGaussians, const float* means, const float* scales, const float* rotations, const float* opacities, const float* colors, const float* harmonics,
                             std::function<uint8_t* (uint64_t size)> createGeomBuf, std::function<uint8_t* (uint64_t size)> createBinningBuf) 
{
	//allocate temp buffers:
	//---------------
	uint8_t* geomBufMem = createGeomBuf(
		MGSDRrenderBuffers::required_mem<MGSDRgeomBuffers>(numGaussians)
	);
	uint8_t* binningBufMem = createBinningBuf(
		MGSDRrenderBuffers::required_mem<MGSDRbinningBuffers>(numGaussians)
	);
	
	MGSDRgeomBuffers geomBufs(geomBufMem, numGaussians);
	MGSDRbinningBuffers binningBufs(binningBufMem, numGaussians);

	//launch preprocess kernel:
	//---------------
	uint32_t numWorkgroupsPreprocess = (numGaussians + MGS_DR_PREPROCESS_WORKGROUP_SIZE - 1) / MGS_DR_PREPROCESS_WORKGROUP_SIZE;

	_mgs_dr_foward_preprocess_kernel<<<numWorkgroupsPreprocess, MGS_DR_PREPROCESS_WORKGROUP_SIZE>>>(
		view, proj, focalX, focalY,
		numGaussians, means, scales, rotations, opacities, colors, harmonics,
		geomBufs.pixCenters, geomBufs.pixRadii, geomBufs.depths, geomBufs.tilesTouched, geomBufs.conic, geomBufs.rgb
	);

	//verifying dummy data
	uint32_t tilesTouched;
	cudaMemcpy(&tilesTouched, geomBufs.tilesTouched, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	printf("tilesTouched: %d\n", tilesTouched);

	//launch render kernel:
	//---------------
	uint32_t numWorkgroupsX = (outWidth  + MGS_DR_TILE_SIZE - 1) / MGS_DR_TILE_SIZE;
	uint32_t numWorkgroupsY = (outHeight + MGS_DR_TILE_SIZE - 1) / MGS_DR_TILE_SIZE;

	_mgs_dr_forward_splat_kernel<<<{ numWorkgroupsX, numWorkgroupsY }, { MGS_DR_TILE_SIZE, MGS_DR_TILE_SIZE }>>>(
		outWidth, outHeight, outImg
	);

	//TODO proper error checking
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
		printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));

	//return:
	//---------------
	return 0;
}

//-------------------------------------------//

__global__ void __launch_bounds__(MGS_DR_PREPROCESS_WORKGROUP_SIZE)
_mgs_dr_foward_preprocess_kernel(const float* view, const float* proj, float focalX, float focalY, 
                                 uint32_t numGaussians, const float* means, const float* scales, const float* rotations, const float* opacities, const float* colors, const float* harmonics,
                                 uint2* outPixCenters, float* outPixRadii, float* outDepths, uint32_t* outTilesTouched, float4* outConic, float3* outRGB)
{
	auto idx = cg::this_grid().thread_rank();
	if(idx >= numGaussians)
		return;

	//writing dummy data for now
	outPixCenters[idx] = {1, 1};
	outPixRadii[idx] = 10.0f;
	outDepths[idx] = 14.0f;
	outTilesTouched[idx] = 33;
	outConic[idx] = { 1.0f, 2.0f, 3.0f, 4.0f };
	outRGB[idx] = { 255.0f, 255.0f, 255.0f };
}

__global__ void __launch_bounds__(MGS_DR_TILE_SIZE * MGS_DR_TILE_SIZE) 
_mgs_dr_forward_splat_kernel(uint32_t width, uint32_t height, float* img)
{
	auto block = cg::this_thread_block();
	uint32_t numBlocksX = (width + MGS_DR_TILE_SIZE - 1) / MGS_DR_TILE_SIZE;

	uint32_t pixelMinX = block.group_index().x * MGS_DR_TILE_SIZE;
	uint32_t pixelMinY = block.group_index().y * MGS_DR_TILE_SIZE;

	uint32_t pixelMaxX = min(pixelMinX + MGS_DR_TILE_SIZE, width );
	uint32_t pixelMaxY = min(pixelMinY + MGS_DR_TILE_SIZE, height);

	uint32_t pixelX = pixelMinX + block.thread_index().x;
	uint32_t pixelY = pixelMinY + block.thread_index().y;
	
	uint32_t pixelId = pixelX + width * pixelY;

	bool inside = pixelX < width && pixelY < height;
	if(!inside)
		return;

	float u = (float)pixelX / width;
	float v = (float)pixelY / height;

	uint32_t writeIdx = pixelId * 4;
	img[writeIdx + 0] = u;
	img[writeIdx + 1] = v;
	img[writeIdx + 2] = 0.0f;
	img[writeIdx + 3] = 1.0f;
}