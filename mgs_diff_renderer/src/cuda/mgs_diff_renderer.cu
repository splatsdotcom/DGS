#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#define MGS_DIFF_RENDERER_WORKGROUP_SIZE 256

//-------------------------------------------//

__global__ void _mgs_diff_render_foward_kernel(const float* in, float* out, int64_t num);

//-------------------------------------------//

extern "C" void mgs_diff_render_forward_cuda(const float* in, float* out, int64_t num) 
{
	if(num == 0) 
		return;
	
	int64_t numWorkgroups = (num + MGS_DIFF_RENDERER_WORKGROUP_SIZE - 1) / MGS_DIFF_RENDERER_WORKGROUP_SIZE;
	_mgs_diff_render_foward_kernel<<<(int)numWorkgroups, MGS_DIFF_RENDERER_WORKGROUP_SIZE>>>(in, out, num);

	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
		printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
}

//-------------------------------------------//

__global__ void _mgs_diff_render_foward_kernel(const float* in, float* out, int64_t num)
{
	int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < num)
		out[idx] = in[idx] * 2.0f;
}