#define __FILENAME__ "mgs_dr_backward.cu"

#include "mgs_dr_backward.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>

#include "mgs_dr_buffers.h"
#include "mgs_dr_global.h"

#define QM_FUNC_ATTRIBS __device__ static inline
#include "../external/QuickMath/quickmath.h"

namespace cg = cooperative_groups;

//-------------------------------------------//

#define MGS_DR_PREPROCESS_WORKGROUP_SIZE 64

//-------------------------------------------//

__global__ static void __launch_bounds__(MGS_DR_TILE_LEN) 
_mgs_dr_backward_splat_kernel(uint32_t width, uint32_t height, const float* dLdImg, const float* transmittances, const uint32_t* numContributors,
                              const uint2* ranges, const uint32_t* indices, const float2* pixCenters, const float4* conic, const float3* rgb,
                              float2* outDLdPixCenters, float4* outDLdConic, float3* outDLdRGB);

__global__ static void __launch_bounds__(MGS_DR_PREPROCESS_WORKGROUP_SIZE)
_mgs_dr_backward_preprocess_kernel(uint32_t width, uint32_t height, const float* view, const float* proj, float focalX, float focalY,
                                   uint32_t numGaussians, const float* means, const float* scales, const float* rotations, const float* opacities, const float* colors, const float* harmonics,
								   const MGSDRcov3D* covs, const float* pixRadii,
                                   float* outDLdMeans, float* outDLdScales, float* outDLdRotations, float* outDLdOpacities, float* outDLdColors, float* outDLdHarmonics);

//-------------------------------------------//

void mgs_dr_backward_cuda(uint32_t width, uint32_t height, const float* dLdImage, const float* view, const float* proj, float focalX, float focalY,
						  uint32_t numGaussians, const float* means, const float* scales, const float* rotations, const float* opacities, const float* colors, const float* harmonics,
						  uint32_t numRendered, const uint8_t* geomBufsMem, const uint8_t* binningBufsMem, const uint8_t* imageBufsMem,
						  float* outDLdMeans, float* outDLdScales, float* outDLdRotations, float* outDLdOpacities, float* outDLdColors, float* outDLdHarmonics,
						  bool debug)
{
	//initialize state from foward pass:
	//---------------
	uint32_t tilesWidth  = _mgs_ceildivide32(width , MGS_DR_TILE_SIZE);
	uint32_t tilesHeight = _mgs_ceildivide32(height, MGS_DR_TILE_SIZE);
	uint32_t tilesLen = tilesWidth * tilesHeight;

	MGSDRgeomBuffers geomBufs = MGSDRgeomBuffers((uint8_t*)geomBufsMem, numGaussians);
	MGSDRbinningBuffers binningBufs = MGSDRbinningBuffers((uint8_t*)binningBufsMem, numRendered);
	MGSDRimageBuffers imageBufs = MGSDRimageBuffers((uint8_t*)imageBufsMem, width * height);

	//allocate memory for intermediate derivatives:
	//---------------
	float2* dLdPixCenters = NULL;
	float4* dLdConic = NULL;
	float3* dLdRGB = NULL;

	cudaMalloc(&dLdPixCenters, numGaussians * sizeof(float2)); //TODO: error checking
	cudaMalloc(&dLdConic     , numGaussians * sizeof(float4));
	cudaMalloc(&dLdRGB       , numGaussians * sizeof(float3));

	cudaMemset(dLdPixCenters, 0, numGaussians * sizeof(float2));
	cudaMemset(dLdConic     , 0, numGaussians * sizeof(float4));
	cudaMemset(dLdRGB       , 0, numGaussians * sizeof(float2));
	
	//backward render:
	//---------------
	_mgs_dr_backward_splat_kernel<<<{ tilesWidth, tilesHeight }, { MGS_DR_TILE_SIZE, MGS_DR_TILE_SIZE }>>>(
		width, height, dLdImage, imageBufs.accumAlpha, imageBufs.numContributors,
		imageBufs.tileRanges, binningBufs.indicesSorted, geomBufs.pixCenters, geomBufs.conic, geomBufs.rgb,
		dLdPixCenters, dLdConic, dLdRGB
	);

	float2 pixCenters[2];
	float4 conic[2];
	float3 rgb[2];

	cudaMemcpy(pixCenters, dLdPixCenters, sizeof(pixCenters), cudaMemcpyDeviceToHost);
	cudaMemcpy(conic, dLdConic, sizeof(conic), cudaMemcpyDeviceToHost);
	cudaMemcpy(rgb, dLdRGB, sizeof(rgb), cudaMemcpyDeviceToHost);

	printf("dL / dPixCenters: [(%f, %f), (%f, %f)]\n", pixCenters[0].x, pixCenters[0].y, pixCenters[1].x, pixCenters[1].y);
	printf("dL / dConic: [(%f, %f, %f, %f), (%f, %f, %f, %f)]\n", conic[0].x, conic[0].y, conic[0].z, conic[0].w, conic[1].x, conic[1].y, conic[1].z, conic[1].w);
	printf("dL / dRGB: [(%f, %f, %f), (%f, %f, %f)]\n", rgb[0].x, rgb[0].y, rgb[0].z, rgb[1].x, rgb[1].y, rgb[1].z);

	//cleanup:
	//---------------
	cudaFree(dLdPixCenters);
	cudaFree(dLdConic);
	cudaFree(dLdRGB);
}

//-------------------------------------------//

__global__ static void __launch_bounds__(MGS_DR_TILE_LEN) 
_mgs_dr_backward_splat_kernel(uint32_t width, uint32_t height, const float* dLdImg, const float* transmittances, const uint32_t* numContributors,
                              const uint2* ranges, const uint32_t* indices, const float2* pixCenters, const float4* conic, const float3* rgb,
                              float2* outDLdPixCenters, float4* outDLdConic, float3* outDLdColor)
{
	//compute pixel position:
	//---------------
	auto block = cg::this_thread_block();
	uint32_t tilesWidth = _mgs_ceildivide32(width, MGS_DR_TILE_SIZE);

	uint32_t pixelMinX = block.group_index().x * MGS_DR_TILE_SIZE;
	uint32_t pixelMinY = block.group_index().y * MGS_DR_TILE_SIZE;

	uint32_t pixelMaxX = min(pixelMinX + MGS_DR_TILE_SIZE, width );
	uint32_t pixelMaxY = min(pixelMinY + MGS_DR_TILE_SIZE, height);

	uint32_t pixelX = pixelMinX + block.thread_index().x;
	uint32_t pixelY = pixelMinY + block.thread_index().y;
	
	uint32_t pixelId = pixelX + width * pixelY;

	bool inside = pixelX < width && pixelY < height;

	//read gaussian range:
	//---------------
	uint2 range = ranges[block.group_index().x + tilesWidth * block.group_index().y];
	int32_t numToRender = range.y - range.x;
	uint32_t numRounds = _mgs_ceildivide32(numToRender, MGS_DR_TILE_LEN);

	//read dLdPixel:
	//---------------
	float dLdPixel[3] = {
		inside ? dLdImg[pixelId * 3 + 0] : 0.0f,
		inside ? dLdImg[pixelId * 3 + 1] : 0.0f,
		inside ? dLdImg[pixelId * 3 + 2] : 0.0f
	};

	float dDxdX = 0.5f * width;
	float dDydY = 0.5f * height;

	//allocate shared memory:
	//---------------
	__shared__ uint32_t collectedIndices[MGS_DR_TILE_LEN];
	__shared__ float2 collectedPixCenters[MGS_DR_TILE_LEN];
	__shared__ float4 collectedConic[MGS_DR_TILE_LEN];
	__shared__ float3 collectedRGB[MGS_DR_TILE_LEN];

	//loop over batches:
	//---------------
	float transmittanceFinal = inside ? transmittances[pixelId] : 0.0f;
	float transmittance = transmittanceFinal;

	uint32_t curContributor = numToRender;
	uint32_t lastContributor = inside ? numContributors[pixelId] : 0;

	float accumColor[3] = {0};
	float lastAlpha = 0.0f;
	float lastColor[3] = {0};

	for(uint32_t i = 0; i < numRounds; i++)
	{
		//sync threads
		block.sync();

		//collectively load gaussian data
		uint32_t loadIdx = i * MGS_DR_TILE_LEN + block.thread_rank();
		if(range.x + loadIdx < range.y)
		{
			//we load back to front
			uint32_t gaussianIdx = indices[range.y - loadIdx - 1];

			collectedIndices[block.thread_rank()] = gaussianIdx;
			collectedPixCenters[block.thread_rank()] = pixCenters[gaussianIdx];
			collectedConic[block.thread_rank()] = conic[gaussianIdx];
			collectedRGB[block.thread_rank()] = rgb[gaussianIdx];
		}

		block.sync();

		//accumulate collected gaussians
		for(uint32_t j = 0; j < min(MGS_DR_TILE_LEN, numToRender); j++)
		{
			curContributor--; //skip if current contributor is ahead of final
			if(curContributor >= lastContributor)
				continue;

			//compute alpha
			uint32_t idx = collectedIndices[j];
			float2 pos = collectedPixCenters[j];
			float4 conic = collectedConic[j];

			float dx = pos.x - (float)pixelX;
			float dy = pos.y - (float)pixelY;

			float power = -0.5f * (conic.x * dx * dx + conic.z * dy * dy) - conic.y * dx * dy;
			if(power > 0.0f)
				continue;

			float G = exp(power);
			float alpha = min(MGS_DR_MAX_ALPHA, conic.w * G);
			if(alpha < MGS_DR_MIN_ALPHA)
				continue;

			//update transmittance
			transmittance /= (1.0f - alpha);

			//compute alpha/color derivs
			const float dChanneldColor = transmittance * alpha;

			float dLdAlpha = 0.0f;
			for(uint32_t k = 0; k < 3; k++)
			{
				float channel = ((float*)&collectedRGB[j])[k];
				float dLdChannel = dLdPixel[k];

				accumColor[k] = lastAlpha * lastColor[k] + (1.0f - lastAlpha) * accumColor[k];
				lastColor[k] = channel;

				dLdAlpha += (channel - accumColor[k]) * dLdChannel;

				atomicAdd((float*)&outDLdColor[idx] + k, dChanneldColor * dLdChannel);
			}

			dLdAlpha *= transmittance;
			lastAlpha = alpha;

			//compute conic derivs
			float dLdG = conic.w * dLdAlpha;
			float dGdDx = -G * dx * conic.x - G * dy * conic.y;
			float dGdDy = -G * dy * conic.z - G * dx * conic.y;
			
			float3 dGdConic = {
				-0.5f * G * dx * dx,
				-0.5f * G * dx * dy, //treating conic.y the off-diag 2x2 covariance matrix, so we mul by 0.5
				-0.5f * G * dy * dy
			};

			//update derivs
			//TODO: make these faster with a shared add first
			atomicAdd(&outDLdPixCenters[idx].x, dLdG * dGdDx * dDxdX);
			atomicAdd(&outDLdPixCenters[idx].y, dLdG * dGdDy * dDydY);

			atomicAdd(&outDLdConic[idx].x, dLdG * dGdConic.x);
			atomicAdd(&outDLdConic[idx].y, dLdG * dGdConic.y);
			atomicAdd(&outDLdConic[idx].z, dLdG * dGdConic.z);

			atomicAdd(&outDLdConic[idx].w, dLdAlpha * G); //opacity
		}

		//decrement num left to render
		numToRender -= MGS_DR_TILE_LEN;
	}
}

__global__ static void __launch_bounds__(MGS_DR_PREPROCESS_WORKGROUP_SIZE)
_mgs_dr_backward_preprocess_kernel(uint32_t width, uint32_t height, const float* view, const float* proj, float focalX, float focalY,
                                   uint32_t numGaussians, const float* means, const float* scales, const float* rotations, const float* opacities, const float* colors, const float* harmonics,
                                   const MGSDRcov3D* covs, const float* pixRadii,
								   const float4* dLdConics,
                                   float* outDLdMeans, float* outDLdScales, float* outDLdRotations, float* outDLdOpacities, float* outDLdColors, float* outDLdHarmonics)
{
	auto idx = cg::this_grid().thread_rank();
	if(idx >= numGaussians || pixRadii[idx] <= 0.0f)
		return;

	//read existing gradients:
	//---------------
	float4 dLdConic = dLdConics[idx];

	//find view pos:
	//---------------
	QMmat4 viewMat = qm_mat4_load(view);
	QMmat4 projMat = qm_mat4_load(proj);
	
	QMvec3 mean = qm_vec3_load(&means[idx * 3]);

	QMvec4 camPos = qm_mat4_mult_vec4(
		viewMat, 
		(QMvec4){ mean.x, mean.y, mean.z, 1.0f }
	);

	//project covariance matrix to 2D:
	//---------------
	QMmat3 cov = {{
		{ covs[idx].m00, covs[idx].m01, covs[idx].m02 },
		{ covs[idx].m10, covs[idx].m11, covs[idx].m12 },
		{ covs[idx].m02, covs[idx].m12, covs[idx].m22 }
	}};
	
	QMmat3 J = {{
		{ -focalX / camPos.z, 0.0,                 (focalX * camPos.x) / (camPos.z * camPos.z) },
		{ 0.0,                -focalY / camPos.z,  (focalY * camPos.y) / (camPos.z * camPos.z) },
		{ 0.0,                0.0,                 0.0                                         }
	}};

	QMmat3 T = qm_mat3_mult(
		qm_mat3_transpose(qm_mat4_top_left(viewMat)),
		J
	);

	QMmat3 cov2d = qm_mat3_mult(
		qm_mat3_transpose(T),
		qm_mat3_mult(cov, T)
	);

	//compute gradients w.r.t. covariance:
	//---------------
	float a = cov2d.m[0][0]; 
	float b = cov2d.m[0][1];
	float c = cov2d.m[1][1];

	float det = a * c - b * b;

	float dLdA, dLdB, dLdC;
	MGSDRcov3D dLdCov;
	if(det != 0.0f)
	{
		float det2Inv = 1.0f / (det * det);
		dLdA =        det2Inv * (-c * c * dLdConic.x + b * c * dLdConic.y + (det -        a * c) * dLdConic.z);
		dLdC =        det2Inv * (-a * a * dLdConic.z + b * a * dLdConic.y + (det -        a * x) * dLdConic.x);
		dLdB = 2.0f * det2Inv * ( b * c * dLdConic.x + a * b * dLdConic.z + (det + 2.0f * b * b) * dLdConic.y);

		// dLdCov.m00 = 
	}
	else
	{
		dLdA = 0.0f;
		dLdC = 0.0f;
		dLdB = 0.0f;
	}
}