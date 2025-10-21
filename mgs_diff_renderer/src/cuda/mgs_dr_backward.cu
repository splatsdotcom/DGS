#define __FILENAME__ "mgs_dr_backward.cu"

#include "mgs_dr_backward.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <cooperative_groups.h>

#include "mgs_dr_buffers.h"
#include "mgs_dr_global.h"

namespace cg = cooperative_groups;

//-------------------------------------------//

#define MGS_DR_PREPROCESS_WORKGROUP_SIZE 64

//-------------------------------------------//

__global__ static void __launch_bounds__(MGS_DR_TILE_LEN) 
_mgs_dr_backward_splat_kernel(uint32_t width, uint32_t height, const float* dLdImg, const float* transmittances, const uint32_t* numContributors,
                              const uint2* ranges, const uint32_t* indices, const MGSDRgeomBuffers geom,
                              MGSDRderivativeBuffers outIntermediate, float* outDLdOpacities);

__global__ static void __launch_bounds__(MGS_DR_PREPROCESS_WORKGROUP_SIZE)
_mgs_dr_backward_preprocess_kernel(uint32_t width, uint32_t height, const float* view, const float* proj, float focalX, float focalY,
                                   uint32_t numGaussians, const float* means, const float* scales, const float* rotations, const float* opacities, const float* colors, const float* harmonics,
								   const MGSDRgeomBuffers geom, const MGSDRderivativeBuffers intermediate,
                                   float* outDLdMeans, float* outDLdScales, float* outDLdRotations, float* outDLdColors, float* outDLdHarmonics);

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

	MGSDRgeomBuffers geomBufs = MGSDRgeomBuffers((uint8_t*)geomBufsMem, numGaussians);
	MGSDRbinningBuffers binningBufs = MGSDRbinningBuffers((uint8_t*)binningBufsMem, numRendered);
	MGSDRimageBuffers imageBufs = MGSDRimageBuffers((uint8_t*)imageBufsMem, width * height);

	//allocate memory for intermediate derivatives:
	//---------------
	uint64_t intermediateDerivMemSize = MGSDRrenderBuffers::required_mem<MGSDRderivativeBuffers>(numGaussians);

	uint8_t* intermediateDerivMem;
	cudaMalloc(&intermediateDerivMem, intermediateDerivMemSize);
	cudaMemset(intermediateDerivMem, 0, intermediateDerivMemSize);

	MGSDRderivativeBuffers intermediateDerivs = MGSDRderivativeBuffers(intermediateDerivMem, numGaussians);

	//backward render:
	//---------------
	_mgs_dr_backward_splat_kernel<<<{ tilesWidth, tilesHeight }, { MGS_DR_TILE_SIZE, MGS_DR_TILE_SIZE }>>>(
		width, height, dLdImage, imageBufs.accumAlpha, imageBufs.numContributors,
		imageBufs.tileRanges, binningBufs.indicesSorted, geomBufs,
		intermediateDerivs, outDLdOpacities
	);

	//backward preprocess:
	//---------------
	uint32_t numWorkgroupsPreprocess = _mgs_ceildivide32(numGaussians, MGS_DR_PREPROCESS_WORKGROUP_SIZE);
	_mgs_dr_backward_preprocess_kernel<<<numWorkgroupsPreprocess, MGS_DR_PREPROCESS_WORKGROUP_SIZE>>>(
		width, height, view, proj, focalX, focalY,
		numGaussians, means, scales, rotations, opacities, colors, harmonics,
		geomBufs, intermediateDerivs,
		outDLdMeans, outDLdScales, outDLdRotations, outDLdColors, outDLdHarmonics
	);

	//cleanup:
	//---------------
	cudaFree(intermediateDerivMem);
}

//-------------------------------------------//

__global__ static void __launch_bounds__(MGS_DR_TILE_LEN) 
_mgs_dr_backward_splat_kernel(uint32_t width, uint32_t height, const float* dLdImg, const float* transmittances, const uint32_t* numContributors,
                              const uint2* ranges, const uint32_t* indices, const MGSDRgeomBuffers geom,
                              MGSDRderivativeBuffers outIntermediate, float* outDLdOpacities)
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
	QMvec3 dLdPixel = {
		inside ? dLdImg[pixelId * 3 + 0] : 0.0f,
		inside ? dLdImg[pixelId * 3 + 1] : 0.0f,
		inside ? dLdImg[pixelId * 3 + 2] : 0.0f
	};

	float dDxdX = 0.5f * width;
	float dDydY = 0.5f * height;

	//allocate shared memory:
	//---------------
	__shared__ uint32_t collectedIndices   [MGS_DR_TILE_LEN];
	__shared__ QMvec2 collectedPixCenters  [MGS_DR_TILE_LEN];
	__shared__ QMvec4 collectedConicOpacity[MGS_DR_TILE_LEN];
	__shared__ QMvec3 collectedRGB         [MGS_DR_TILE_LEN];

	//loop over batches:
	//---------------
	float transmittanceFinal = inside ? transmittances[pixelId] : 0.0f;
	float transmittance = transmittanceFinal;

	uint32_t curContributor = numToRender;
	uint32_t lastContributor = inside ? numContributors[pixelId] : 0;

	QMvec3 accumColor = qm_vec3_full(0.0f);
	float lastAlpha = 0.0f;
	QMvec3 lastColor = qm_vec3_full(0.0f);

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

			collectedIndices     [block.thread_rank()] = gaussianIdx;
			collectedPixCenters  [block.thread_rank()] = geom.pixCenters[gaussianIdx];
			collectedConicOpacity[block.thread_rank()] = geom.conicOpacity[gaussianIdx];
			collectedRGB         [block.thread_rank()] = geom.rgb[gaussianIdx];
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
			QMvec2 pos = collectedPixCenters[j];
			QMvec4 conicO = collectedConicOpacity[j];

			float dx = pos.x - (float)pixelX;
			float dy = pos.y - (float)pixelY;

			float power = -0.5f * (conicO.x * dx * dx + conicO.z * dy * dy) - conicO.y * dx * dy;
			if(power > 0.0f)
				continue;

			float G = exp(power);
			float alpha = min(MGS_DR_MAX_ALPHA, conicO.w * G);
			if(alpha < MGS_DR_MIN_ALPHA)
				continue;

			//update transmittance
			transmittance /= (1.0f - alpha);

			//compute alpha/color derivs
			const float dChanneldColor = transmittance * alpha;

			float dLdAlpha = 0.0f;
			for(uint32_t k = 0; k < 3; k++)
			{
				float channel = collectedRGB[j].v[k];
				float dLdChannel = dLdPixel.v[k];

				accumColor.v[k] = lastAlpha * lastColor.v[k] + (1.0f - lastAlpha) * accumColor.v[k];
				lastColor.v[k] = channel;

				dLdAlpha += (channel - accumColor.v[k]) * dLdChannel;

				atomicAdd(&outIntermediate.dLdColors[idx].v[k], dChanneldColor * dLdChannel);
			}

			dLdAlpha *= transmittance;
			lastAlpha = alpha;

			//compute conic derivs
			float dLdG = conicO.w * dLdAlpha;
			float dGdDx = -G * dx * conicO.x - G * dy * conicO.y;
			float dGdDy = -G * dy * conicO.z - G * dx * conicO.y;
			
			QMvec3 dGdConic = {
				-0.5f * G * dx * dx,
				-0.5f * G * dx * dy, //treating conic.y the off-diag 2x2 covariance matrix, so we mul by 0.5
				-0.5f * G * dy * dy
			};

			//update derivs
			//TODO: make these faster with a shared add first
			atomicAdd(&outIntermediate.dLdPixCenters[idx].x, dLdG * dGdDx * dDxdX);
			atomicAdd(&outIntermediate.dLdPixCenters[idx].y, dLdG * dGdDy * dDydY);

			atomicAdd(&outIntermediate.dLdConics[idx].x, dLdG * dGdConic.x);
			atomicAdd(&outIntermediate.dLdConics[idx].y, dLdG * dGdConic.y);
			atomicAdd(&outIntermediate.dLdConics[idx].z, dLdG * dGdConic.z);

			atomicAdd(&outDLdOpacities[idx], dLdAlpha * G);
		}

		//decrement num left to render
		numToRender -= MGS_DR_TILE_LEN;
	}
}

__global__ static void __launch_bounds__(MGS_DR_PREPROCESS_WORKGROUP_SIZE)
_mgs_dr_backward_preprocess_kernel(uint32_t width, uint32_t height, const float* view, const float* proj, float focalX, float focalY,
                                   uint32_t numGaussians, const float* means, const float* scales, const float* rotations, const float* opacities, const float* colors, const float* harmonics,
                                   const MGSDRgeomBuffers geom, const MGSDRderivativeBuffers intermediate,
                                   float* outDLdMeans, float* outDLdScales, float* outDLdRotations, float* outDLdColors, float* outDLdHarmonics)
{
	auto idx = cg::this_grid().thread_rank();
	if(idx >= numGaussians || geom.pixRadii[idx] <= 0.0f)
		return;

	//find view + clip pos:
	//---------------
	QMmat4 viewMat = qm_mat4_load(view);
	QMmat4 projMat = qm_mat4_load(proj);
	
	QMvec3 mean = qm_vec3_load(&means[idx * 3]);

	QMvec4 camPos = qm_mat4_mult_vec4(
		viewMat, 
		(QMvec4){ mean.x, mean.y, mean.z, 1.0f }
	);

	QMvec4 clipPos = qm_mat4_mult_vec4(
		projMat, camPos
	);

	//project covariance matrix to 2D:
	//---------------
	QMmat3 cov = {{
		{ geom.covs[idx].m00, geom.covs[idx].m01, geom.covs[idx].m02 },
		{ geom.covs[idx].m01, geom.covs[idx].m11, geom.covs[idx].m12 },
		{ geom.covs[idx].m02, geom.covs[idx].m12, geom.covs[idx].m22 }
	}};
	
	QMmat3 J = {{
		{ -focalX / camPos.z, 0.0,                 (focalX * camPos.x) / (camPos.z * camPos.z) },
		{ 0.0,                -focalY / camPos.z,  (focalY * camPos.y) / (camPos.z * camPos.z) },
		{ 0.0,                0.0,                 0.0                                         }
	}};

	QMmat3 W = qm_mat3_transpose(qm_mat4_top_left(viewMat));
	QMmat3 T = qm_mat3_mult(W, J);

	QMmat3 cov2d = qm_mat3_mult(
		qm_mat3_transpose(T),
		qm_mat3_mult(cov, T)
	);

	//compute gradients w.r.t. covariance:
	//---------------
	QMvec3 dLdConic = intermediate.dLdConics[idx];

	float a = cov2d.m[0][0]; 
	float b = cov2d.m[0][1];
	float c = cov2d.m[1][1];

	float det = a * c - b * b;

	float dLdA, dLdB, dLdC;
	MGSDRcov3D dLdCov;
	if(det != 0.0f)
	{
		float det2Inv = 1.0f / (det * det);
		dLdA =        det2Inv * (-c * c * dLdConic.x + 2.0f * b * c         * dLdConic.y + (det - a * c) * dLdConic.z);
		dLdC =        det2Inv * (-a * a * dLdConic.z + 2.0f * b * a         * dLdConic.y + (det - a * c) * dLdConic.x);
		dLdB = 2.0f * det2Inv * ( b * c * dLdConic.x - (det + 2.0f * b * b) * dLdConic.y + a * b         * dLdConic.z);

		dLdCov.m00 = T.m[0][0] * T.m[0][0] * dLdA + T.m[0][0] * T.m[1][0] * dLdB + T.m[1][0] * T.m[1][0] * dLdC;
		dLdCov.m11 = T.m[0][1] * T.m[0][1] * dLdA + T.m[0][1] * T.m[1][1] * dLdB + T.m[1][1] * T.m[1][1] * dLdC;
		dLdCov.m22 = T.m[0][2] * T.m[0][2] * dLdA + T.m[0][2] * T.m[1][2] * dLdB + T.m[1][2] * T.m[1][2] * dLdC;

		dLdCov.m01 = 2.0f * T.m[0][0] * T.m[0][1] * dLdA + (T.m[0][0] * T.m[1][1] + T.m[0][1] * T.m[1][0]) * dLdB + 2.0f * T.m[1][0] * T.m[1][1] * dLdC;
		dLdCov.m02 = 2.0f * T.m[0][0] * T.m[0][2] * dLdA + (T.m[0][0] * T.m[1][2] + T.m[0][2] * T.m[1][0]) * dLdB + 2.0f * T.m[1][0] * T.m[1][2] * dLdC;
		dLdCov.m12 = 2.0f * T.m[0][2] * T.m[0][1] * dLdA + (T.m[0][1] * T.m[1][2] + T.m[0][2] * T.m[1][1]) * dLdB + 2.0f * T.m[1][1] * T.m[1][2] * dLdC;
	}
	else
	{
		dLdA = 0.0f;
		dLdC = 0.0f;
		dLdB = 0.0f;

		dLdCov = (MGSDRcov3D){0};
	}

	//compute gradients w.r.t. mean (where the mean affects the jacobian):
	//---------------
	float dLdT00 = 2.0f * (T.m[0][0] * cov.m[0][0] + T.m[0][1] * cov.m[0][1] + T.m[0][2] * cov.m[0][2]) * dLdA +
	                      (T.m[1][0] * cov.m[0][0] + T.m[1][1] * cov.m[0][1] + T.m[1][2] * cov.m[0][2]) * dLdB;
	float dLdT01 = 2.0f * (T.m[0][0] * cov.m[1][0] + T.m[0][1] * cov.m[1][1] + T.m[0][2] * cov.m[1][2]) * dLdA +
	                      (T.m[1][0] * cov.m[1][0] + T.m[1][1] * cov.m[1][1] + T.m[1][2] * cov.m[1][2]) * dLdB;
	float dLdT02 = 2.0f * (T.m[0][0] * cov.m[2][0] + T.m[0][1] * cov.m[2][1] + T.m[0][2] * cov.m[2][2]) * dLdA +
	                      (T.m[1][0] * cov.m[2][0] + T.m[1][1] * cov.m[2][1] + T.m[1][2] * cov.m[2][2]) * dLdB;
	float dLdT10 = 2.0f * (T.m[1][0] * cov.m[0][0] + T.m[1][1] * cov.m[0][1] + T.m[1][2] * cov.m[0][2]) * dLdC +
	                      (T.m[0][0] * cov.m[0][0] + T.m[0][1] * cov.m[0][1] + T.m[0][2] * cov.m[0][2]) * dLdB;
	float dLdT11 = 2.0f * (T.m[1][0] * cov.m[1][0] + T.m[1][1] * cov.m[1][1] + T.m[1][2] * cov.m[1][2]) * dLdC +
	                      (T.m[0][0] * cov.m[1][0] + T.m[0][1] * cov.m[1][1] + T.m[0][2] * cov.m[1][2]) * dLdB;
	float dLdT12 = 2.0f * (T.m[1][0] * cov.m[2][0] + T.m[1][1] * cov.m[2][1] + T.m[1][2] * cov.m[2][2]) * dLdC +
	                      (T.m[0][0] * cov.m[2][0] + T.m[0][1] * cov.m[2][1] + T.m[0][2] * cov.m[2][2]) * dLdB;

	float dLdJ00 = W.m[0][0] * dLdT00 + W.m[0][1] * dLdT01 + W.m[0][2] * dLdT02;
	float dLdJ02 = W.m[2][0] * dLdT00 + W.m[2][1] * dLdT01 + W.m[2][2] * dLdT02;
	float dLdJ11 = W.m[1][0] * dLdT10 + W.m[1][1] * dLdT11 + W.m[1][2] * dLdT12;
	float dLdJ12 = W.m[2][0] * dLdT10 + W.m[2][1] * dLdT11 + W.m[2][2] * dLdT12;

	float invCamZ = 1.0f / camPos.z;
	float invCamZ2 = invCamZ * invCamZ;
	float invCamZ3 = invCamZ * invCamZ * invCamZ;

	float dLdCamPosX = focalX * invCamZ2 * dLdJ02;
	float dLdCamPosY = focalY * invCamZ2 * dLdJ12;
	float dLdCamPosZ = focalX * invCamZ2 * dLdJ00 + focalY * invCamZ2 * dLdJ11 - (2.0f * focalX * camPos.x) * invCamZ3 * dLdJ02 - (2.0f * focalY * camPos.y) * invCamZ3 * dLdJ12;

	//this is only part of dLdMean (how it affects the jacobian), dLdMean w.r.t. dLdPixCenters will be computed later
	QMvec4 dLdMeanJ = qm_mat4_mult_vec4(
		qm_mat4_transpose(viewMat),
		(QMvec4){ (float)dLdCamPosX, (float)dLdCamPosY, (float)dLdCamPosZ, 0.0f } 
	);

	//compute gradients w.r.t. mean (where the mean affects screenspace position):
	//---------------
	QMmat4 PV = qm_mat4_mult(projMat, viewMat);

	float clipW = 1.0f / (clipPos.w + 0.000001f);
	float mul1 = (PV.m[0][0] * mean.x + PV.m[1][0] * mean.y + PV.m[2][0] * mean.z + PV.m[3][0]) * clipW * clipW;
	float mul2 = (PV.m[0][1] * mean.x + PV.m[1][1] * mean.y + PV.m[2][1] * mean.z + PV.m[3][1]) * clipW * clipW;
	
	QMvec3 dLdMeanScreenspace;
	dLdMeanScreenspace.x = (PV.m[0][0] * clipW - PV.m[0][3] * mul1) * intermediate.dLdPixCenters[idx].x + (PV.m[0][1] * clipW - PV.m[0][3] * mul2) * intermediate.dLdPixCenters[idx].y;
	dLdMeanScreenspace.y = (PV.m[1][4] * clipW - PV.m[1][3] * mul1) * intermediate.dLdPixCenters[idx].x + (PV.m[1][1] * clipW - PV.m[1][3] * mul2) * intermediate.dLdPixCenters[idx].y;
	dLdMeanScreenspace.z = (PV.m[2][0] * clipW - PV.m[2][3] * mul1) * intermediate.dLdPixCenters[idx].x + (PV.m[2][1] * clipW - PV.m[2][3] * mul2) * intermediate.dLdPixCenters[idx].y;

	QMvec3 dLdMean = {
		dLdMeanScreenspace.x + dLdMeanJ.x,
		dLdMeanScreenspace.y + dLdMeanJ.y,
		dLdMeanScreenspace.z + dLdMeanJ.z
	};

	//compute gradients w.r.t. scale and rotation:
	//---------------
	QMvec3 scale = qm_vec3_load(&scales[idx * 3]);
	QMquaternion rot = qm_quaternion_load(&rotations[idx * 4]);

	QMmat3 scaleMat = qm_mat4_top_left(qm_mat4_scale(scale));
	QMmat3 rotMat = qm_mat4_top_left(qm_quaternion_to_mat4(rot));

	QMmat3 M = qm_mat3_mult(scaleMat, rotMat);
	QMmat3 dLdM = qm_mat3_mult(
		M, (QMmat3){{
			{ 2.0f * dLdCov.m00,        dLdCov.m01,        dLdCov.m02 },
			{        dLdCov.m01, 2.0f * dLdCov.m11,        dLdCov.m12 },
			{        dLdCov.m02,        dLdCov.m12, 2.0f * dLdCov.m22 }
		}}
	);

	QMmat3 rotMatT = qm_mat3_transpose(rotMat);
	QMmat3 dLdMT = qm_mat3_transpose(dLdM);

	QMvec3 dLdScale = {
		qm_vec3_dot(rotMatT.v[0], dLdMT.v[0]),
		qm_vec3_dot(rotMatT.v[1], dLdMT.v[1]),
		qm_vec3_dot(rotMatT.v[2], dLdMT.v[2])
	};

	for(uint32_t i = 0; i < 3; i++)
		dLdMT.v[i] = qm_vec3_scale(dLdMT.v[i], scale.v[i]);

	QMquaternion dLdRot;
	dLdRot.x = 2.0f * rot.y * (dLdMT.m[1][0] + dLdMT.m[0][1]) + 2.0f * rot.z * (dLdMT.m[2][0] + dLdMT.m[0][2]) + 2.0f * rot.w * (dLdMT.m[1][2] - dLdMT.m[2][1]) - 4.0f * rot.x * (dLdMT.m[2][2] + dLdMT.m[1][1]);
	dLdRot.y = 2.0f * rot.x * (dLdMT.m[1][0] + dLdMT.m[0][1]) + 2.0f * rot.w * (dLdMT.m[2][0] - dLdMT.m[0][2]) + 2.0f * rot.z * (dLdMT.m[1][2] + dLdMT.m[2][1]) - 4.0f * rot.y * (dLdMT.m[2][2] + dLdMT.m[0][0]);
	dLdRot.z = 2.0f * rot.w * (dLdMT.m[1][0] - dLdMT.m[0][1]) + 2.0f * rot.x * (dLdMT.m[2][0] + dLdMT.m[0][2]) + 2.0f * rot.y * (dLdMT.m[2][1] + dLdMT.m[1][2]) - 4.0f * rot.z * (dLdMT.m[0][0] + dLdMT.m[1][1]);
	dLdRot.w = 2.0f * rot.z * (dLdMT.m[0][1] - dLdMT.m[1][0]) + 2.0f * rot.y * (dLdMT.m[2][0] - dLdMT.m[0][2]) + 2.0f * rot.x * (dLdMT.m[1][2] - dLdMT.m[2][1]);

	//compute gradients w.r.t. harmonics:
	//---------------
	QMvec3 dLdColor = intermediate.dLdColors[idx];

	//TODO

	//write out:
	//---------------
	qm_vec3_store(dLdMean, &outDLdMeans[idx * 3]);
	qm_vec3_store(dLdScale, &outDLdScales[idx * 3]);
	qm_quaternion_store(dLdRot, &outDLdRotations[idx * 4]);
	qm_vec3_store(dLdColor, &outDLdColors[idx * 3]);
}