#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define MGS_DIFF_RENDERER_WORKGROUP_SIZE 32

//-------------------------------------------//

__global__ void __launch_bounds__(MGS_DIFF_RENDERER_WORKGROUP_SIZE * MGS_DIFF_RENDERER_WORKGROUP_SIZE) _mgs_diff_render_foward_kernel(uint32_t width, uint32_t height, float* img);

//-------------------------------------------//

extern "C" void mgs_diff_render_forward_cuda(uint32_t outWidth, uint32_t outHeight, float* outImg, const float* view, const float* proj, float focalX, float focalY,
                                             const float* means, const float* scales, const float* rotations, const float* opacities, const float* colors, const float* harmonics) 
{
	uint32_t numWorkgroupsX = (outWidth  + MGS_DIFF_RENDERER_WORKGROUP_SIZE - 1) / MGS_DIFF_RENDERER_WORKGROUP_SIZE;
	uint32_t numWorkgroupsY = (outHeight + MGS_DIFF_RENDERER_WORKGROUP_SIZE - 1) / MGS_DIFF_RENDERER_WORKGROUP_SIZE;

	_mgs_diff_render_foward_kernel<<<{ numWorkgroupsX, numWorkgroupsY }, { MGS_DIFF_RENDERER_WORKGROUP_SIZE, MGS_DIFF_RENDERER_WORKGROUP_SIZE }>>>(
		outWidth, outHeight, outImg
	);

	//TODO proper error checking
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
		printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
}

//-------------------------------------------//

__global__ void __launch_bounds__(MGS_DIFF_RENDERER_WORKGROUP_SIZE * MGS_DIFF_RENDERER_WORKGROUP_SIZE) _mgs_diff_render_foward_kernel(uint32_t width, uint32_t height, float* img)
{
	auto block = cg::this_thread_block();
	uint32_t numBlocksX = (width + MGS_DIFF_RENDERER_WORKGROUP_SIZE - 1) / MGS_DIFF_RENDERER_WORKGROUP_SIZE;

	uint32_t pixelMinX = block.group_index().x * MGS_DIFF_RENDERER_WORKGROUP_SIZE;
	uint32_t pixelMinY = block.group_index().y * MGS_DIFF_RENDERER_WORKGROUP_SIZE;

	uint32_t pixelMaxX = min(pixelMinX + MGS_DIFF_RENDERER_WORKGROUP_SIZE, width );
	uint32_t pixelMaxY = min(pixelMinY + MGS_DIFF_RENDERER_WORKGROUP_SIZE, height);

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