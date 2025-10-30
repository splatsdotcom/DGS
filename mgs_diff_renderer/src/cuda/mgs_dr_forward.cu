#define __FILENAME__ "mgs_dr_forward.cu"

#include "mgs_dr_forward.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include "mgs_dr_buffers.h"
#include "mgs_dr_global.h"

namespace cg = cooperative_groups;

//-------------------------------------------//

//TODO: tweak these!

#define MGS_DR_PREPROCESS_WORKGROUP_SIZE       64
#define MGS_DR_KEY_WRITE_WORKGROUP_SIZE        64
#define MGS_DR_FIND_TILE_RANGES_WORKGROUP_SIZE 64

//-------------------------------------------//

__global__ static void __launch_bounds__(MGS_DR_PREPROCESS_WORKGROUP_SIZE)
_mgs_dr_foward_preprocess_kernel(MGSDRsettings settings, MGSDRgaussians gaussians, MGSDRgeomBuffers outGeom);

__global__ static void __launch_bounds__(MGS_DR_KEY_WRITE_WORKGROUP_SIZE)
_mgs_dr_forward_write_keys_kernel(uint32_t width, uint32_t height,
                                  uint32_t numGaussians, const MGSDRgeomBuffers geom,
                                  uint64_t* outKeys, uint32_t* outValues);

__global__ static void __launch_bounds__(MGS_DR_FIND_TILE_RANGES_WORKGROUP_SIZE)
_mgs_dr_forward_find_tile_ranges_kernel(uint32_t numRendered, const uint64_t* keys, uint2* outRanges);

__global__ static void __launch_bounds__(MGS_DR_TILE_LEN) 
_mgs_dr_forward_splat_kernel(MGSDRsettings settings, 
                             const uint2* ranges, const uint32_t* indices, const MGSDRgeomBuffers geom,
                             float* outColor, float* outAccumAlpha, uint32_t* outNumContributors);

__device__ static void _mgs_dr_get_tile_bounds(uint32_t width, uint32_t height, QMvec2 pixCenter, float pixRadius, uint2& tileMin, uint2& tileMax);

//-------------------------------------------//

uint32_t mgs_dr_forward_cuda(MGSDRsettings settings, MGSDRgaussians gaussians,
                             MGSDRresizeFunc createGeomBuf, MGSDRresizeFunc createBinningBuf, MGSDRresizeFunc createImageBuf,
                             float* outImg)
{
	//validate:
	//---------------
	if(gaussians.count == 0)
		return 0;

	//start profiling:
	//---------------
	MGS_DR_PROFILE_REGION_START(total);

	//allocate geometry + image buffers:
	//---------------
	MGS_DR_PROFILE_REGION_START(allocateGeomImage);

	uint32_t tilesWidth  = _mgs_ceildivide32(settings.width , MGS_DR_TILE_SIZE);
	uint32_t tilesHeight = _mgs_ceildivide32(settings.height, MGS_DR_TILE_SIZE);
	uint32_t tilesLen = tilesWidth * tilesHeight;

	uint8_t* geomBufMem = createGeomBuf(
		MGSDRrenderBuffers::required_mem<MGSDRgeomBuffers>(gaussians.count)
	);
	uint8_t* imageBufMem = createImageBuf(
		MGSDRimageBuffers::required_mem<MGSDRimageBuffers>(settings.width * settings.height)
	);
	
	MGSDRgeomBuffers geomBufs = MGS_DR_CUDA_ERROR_CHECK(MGSDRgeomBuffers(
		geomBufMem, gaussians.count
	));
	MGSDRimageBuffers imageBufs = MGS_DR_CUDA_ERROR_CHECK(MGSDRimageBuffers(
		imageBufMem, settings.width * settings.height
	));

	MGS_DR_PROFILE_REGION_END(allocateGeomImage);

	//preprocess:
	//---------------
	MGS_DR_PROFILE_REGION_START(preprocess);

	uint32_t numWorkgroupsPreprocess = _mgs_ceildivide32(gaussians.count, MGS_DR_PREPROCESS_WORKGROUP_SIZE);
	_mgs_dr_foward_preprocess_kernel<<<numWorkgroupsPreprocess, MGS_DR_PREPROCESS_WORKGROUP_SIZE>>>(
		settings, gaussians, geomBufs
	);
	MGS_DR_CUDA_ERROR_CHECK();

	MGS_DR_PROFILE_REGION_END(preprocess);

	//prefix sum on tile counts:
	//---------------
	MGS_DR_PROFILE_REGION_START(tileCountScan);

	MGS_DR_CUDA_ERROR_CHECK(cub::DeviceScan::InclusiveSum(
		geomBufs.tilesTouchedScanTemp, geomBufs.tilesTouchedScanTempSize, geomBufs.tilesTouched, geomBufs.tilesTouchedScan, gaussians.count
	));

	uint32_t numRendered;
	MGS_DR_CUDA_ERROR_CHECK(cudaMemcpy(
		&numRendered, geomBufs.tilesTouchedScan + gaussians.count - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost
	));

	MGS_DR_PROFILE_REGION_END(tileCountScan);

	if(numRendered == 0)
		return 0;

	//allocate binning buffers:
	//---------------
	MGS_DR_PROFILE_REGION_START(allocateBinning);

	uint8_t* binningBufMem = createBinningBuf(
		MGSDRrenderBuffers::required_mem<MGSDRbinningBuffers>(numRendered)
	);

	MGSDRbinningBuffers binningBufs = MGS_DR_CUDA_ERROR_CHECK(MGSDRbinningBuffers(
		binningBufMem, numRendered
	));

	MGS_DR_PROFILE_REGION_END(allocateBinning);

	//write keys:
	//---------------
	MGS_DR_PROFILE_REGION_START(writeKeys);

	uint32_t numWorkgroupsWriteKeys = _mgs_ceildivide32(gaussians.count, MGS_DR_KEY_WRITE_WORKGROUP_SIZE);
	_mgs_dr_forward_write_keys_kernel<<<numWorkgroupsWriteKeys, MGS_DR_KEY_WRITE_WORKGROUP_SIZE>>>(
		settings.width, settings.height,
		gaussians.count, geomBufs,
		binningBufs.keys, binningBufs.indices
	);
	MGS_DR_CUDA_ERROR_CHECK();

	MGS_DR_PROFILE_REGION_END(writeKeys);

	//sort keys:
	//---------------
	MGS_DR_PROFILE_REGION_START(sortKeys);

	uint32_t numTileBits = 0;
	while(tilesLen > 0)
	{
		numTileBits++;
		tilesLen >>= 1;
	}

	MGS_DR_CUDA_ERROR_CHECK(cub::DeviceRadixSort::SortPairs(
		binningBufs.sortTemp, binningBufs.sortTempSize,
		binningBufs.keys, binningBufs.keysSorted, binningBufs.indices, binningBufs.indicesSorted,
		numRendered, 0, 32 + numTileBits
	));

	MGS_DR_PROFILE_REGION_END(sortKeys);

	//get tile ranges:
	//---------------
	MGS_DR_PROFILE_REGION_START(tileRanges);

	MGS_DR_CUDA_ERROR_CHECK(cudaMemset(
		imageBufs.tileRanges, 0, tilesWidth * tilesHeight * sizeof(uint2)
	));

	uint32_t numWorkgroupsFindTileRanges = _mgs_ceildivide32(numRendered, MGS_DR_FIND_TILE_RANGES_WORKGROUP_SIZE);
	_mgs_dr_forward_find_tile_ranges_kernel<<<numWorkgroupsFindTileRanges, MGS_DR_FIND_TILE_RANGES_WORKGROUP_SIZE>>>(
		numRendered, binningBufs.keysSorted, imageBufs.tileRanges
	);
	MGS_DR_CUDA_ERROR_CHECK();

	MGS_DR_PROFILE_REGION_END(tileRanges);

	//splat:
	//---------------
	MGS_DR_PROFILE_REGION_START(splat);

	_mgs_dr_forward_splat_kernel<<<{ tilesWidth, tilesHeight }, { MGS_DR_TILE_SIZE, MGS_DR_TILE_SIZE }>>>(
		settings,
		imageBufs.tileRanges, binningBufs.indicesSorted, geomBufs,
		outImg, imageBufs.accumAlpha, imageBufs.numContributors
	);
	MGS_DR_CUDA_ERROR_CHECK();

	MGS_DR_PROFILE_REGION_END(splat);

	//print timing information:
	//---------------
	MGS_DR_PROFILE_REGION_END(total);

#ifdef MGS_DR_PROFILE
	std::cout << std::endl;
	std::cout << "TOTAL FRAME TIME (forwards): " << MGS_DR_PROFILE_REGION_TIME(total) << "ms" << std::endl;
	std::cout << "\t- Allocating geom + image buffers: " << MGS_DR_PROFILE_REGION_TIME(allocateGeomImage) << "ms" << std::endl;
	std::cout << "\t- Preprocessing:                   " << MGS_DR_PROFILE_REGION_TIME(preprocess)        << "ms" << std::endl;
	std::cout << "\t- Tile count scan:                 " << MGS_DR_PROFILE_REGION_TIME(tileCountScan)     << "ms" << std::endl;
	std::cout << "\t- Allocating binning buffers:      " << MGS_DR_PROFILE_REGION_TIME(allocateBinning)   << "ms" << std::endl;
	std::cout << "\t- Writing render keys:             " << MGS_DR_PROFILE_REGION_TIME(writeKeys)         << "ms" << std::endl;
	std::cout << "\t- Sorting render keys:             " << MGS_DR_PROFILE_REGION_TIME(sortKeys)          << "ms" << std::endl;
	std::cout << "\t- Finding tile ranges:             " << MGS_DR_PROFILE_REGION_TIME(tileRanges)        << "ms" << std::endl;
	std::cout << "\t- Rasterizing:                     " << MGS_DR_PROFILE_REGION_TIME(splat)             << "ms" << std::endl;
	std::cout << std::endl;
#endif

	//return:
	//---------------
	return numRendered;
}

//-------------------------------------------//

__global__ static void __launch_bounds__(MGS_DR_PREPROCESS_WORKGROUP_SIZE)
_mgs_dr_foward_preprocess_kernel(MGSDRsettings settings, MGSDRgaussians gaussians, MGSDRgeomBuffers outGeom)
{
	auto idx = cg::this_grid().thread_rank();
	if(idx >= gaussians.count)
		return;

	outGeom.tilesTouched[idx] = 0; //so we dont render if culled
	outGeom.pixRadii[idx] = 0.0f;

	//find view and clip pos:
	//---------------
	QMvec3 mean = qm_vec3_load(&gaussians.means[idx * 3]);

	QMvec4 camPos = qm_mat4_mult_vec4(
		settings.view, 
		(QMvec4){ mean.x, mean.y, mean.z, 1.0f }
	);
	QMvec4 clipPos = qm_mat4_mult_vec4(
		settings.proj, camPos
	);

	//cull gaussians out of view:
	//---------------

	//TODO: tweak
	float clip = 1.2 * clipPos.w;
	if(clipPos.x >  clip || clipPos.y >  clip || clipPos.z >  clip ||
	   clipPos.x < -clip || clipPos.y < -clip || clipPos.z < -clip)
		return;

	//compute covariance matrix:
	//---------------
	QMmat4 scaleMat = qm_mat4_scale(qm_vec3_load(&gaussians.scales[idx * 3]));
	QMmat4 rotMat = qm_quaternion_to_mat4(qm_quaternion_load(&gaussians.rotations[idx * 4]));

	//TODO: add mat3 scale and rot functions to QM so we dont need to do a top_left
	QMmat3 M = qm_mat4_top_left(qm_mat4_mult(scaleMat, rotMat));
	QMmat3 cov = qm_mat3_mult(qm_mat3_transpose(M), M);

	//project covariance matrix to 2D:
	//---------------
	QMmat3 J = {{
		{ -settings.focalX / camPos.z, 0.0,                         (settings.focalX * camPos.x) / (camPos.z * camPos.z) },
		{ 0.0,                         -settings.focalY / camPos.z, (settings.focalY * camPos.y) / (camPos.z * camPos.z) },
		{ 0.0,                         0.0,                         0.0                                                  }
	}};

	QMmat3 W = qm_mat3_transpose(qm_mat4_top_left(settings.view));
	QMmat3 T = qm_mat3_mult(W, J);

	QMmat3 cov2d = qm_mat3_mult(
		qm_mat3_transpose(T),
		qm_mat3_mult(cov, T)
	);

	//compute inverse 2d covariance:
	//---------------
	float det = cov2d.m[0][0] * cov2d.m[1][1] - cov2d.m[0][1] * cov2d.m[0][1];
	if(det == 0.0f)
		return;

	QMvec3 conic = qm_vec3_scale((QMvec3){ cov2d.m[1][1], -cov2d.m[0][1], cov2d.m[0][0] }, 1.0f / det);

	//compute eigenvalues:
	//---------------
	float midpoint = (cov2d.m[0][0] + cov2d.m[1][1]) / 2.0;
	float radius = qm_vec2_length((QMvec2){ (cov2d.m[0][0] - cov2d.m[1][1]) / 2.0, cov2d.m[0][1] });

	float lambda1 = midpoint + radius;
	float lambda2 = midpoint - radius;

	//compute image tiles:
	//---------------
	QMvec2 pixCenter = {
		((clipPos.x / clipPos.w + 1.0f) * 0.5f * settings.width ) - 0.5f,
		((clipPos.y / clipPos.w + 1.0f) * 0.5f * settings.height) - 0.5f
	};

	//TODO: tweak
	float pixRadius = ceil(3.0f * sqrt(max(lambda1, lambda2)));

	uint2 tilesMin, tilesMax;
	_mgs_dr_get_tile_bounds(
		settings.width, settings.height, pixCenter, pixRadius,
		tilesMin, tilesMax
	);

	if(tilesMin.x >= tilesMax.x || tilesMin.y >= tilesMax.y)
		return;

	//compute spherical harmonics:
	//---------------
	QMvec3 color = qm_vec3_load(&gaussians.harmonics[idx * 3]);

	//TODO: actual SH

	//write out:
	//---------------
	outGeom.pixCenters  [idx] = { pixCenter.x, pixCenter.y };
	outGeom.pixRadii    [idx] = pixRadius;
	outGeom.depths      [idx] = camPos.z;
	outGeom.tilesTouched[idx] = (tilesMax.x - tilesMin.x) * (tilesMax.y - tilesMin.y);
	outGeom.covs        [idx] = { cov.m[0][0], cov.m[1][0], cov.m[2][0], cov.m[1][1], cov.m[2][1], cov.m[2][2] };
	outGeom.conicOpacity[idx] = { conic.x, conic.y, conic.z, gaussians.opacities[idx] };
	outGeom.color       [idx] = { color.x, color.y, color.z };
}

__global__ static void __launch_bounds__(MGS_DR_KEY_WRITE_WORKGROUP_SIZE)
_mgs_dr_forward_write_keys_kernel(uint32_t width, uint32_t height,
                                  uint32_t numGaussians, const MGSDRgeomBuffers geom,
                                  uint64_t* outKeys, uint32_t* outValues)
{
	auto idx = cg::this_grid().thread_rank();
	if(idx >= numGaussians)
		return;

	//skip if gaussian is not visible:
	//---------------
	if(geom.pixRadii[idx] == 0.0f)
		return;

	//get tile bounds:
	//---------------
	uint2 tilesMin, tilesMax;
	_mgs_dr_get_tile_bounds(
		width, height, geom.pixCenters[idx], geom.pixRadii[idx],
		tilesMin, tilesMax
	);

	//write keys:
	//---------------
	uint32_t writeIdx = (idx == 0) ? 0 : geom.tilesTouchedScan[idx - 1];
	uint32_t tilesWidth  = _mgs_ceildivide32(width , MGS_DR_TILE_SIZE);

	for(uint32_t y = tilesMin.y; y < tilesMax.y; y++)
	for(uint32_t x = tilesMin.x; x < tilesMax.x; x++)
	{
		uint32_t tileIdx = x + tilesWidth * y;
		uint64_t key = ((uint64_t)tileIdx << 32) | *((uint32_t*)&geom.depths[idx]);

		outKeys[writeIdx] = key;
		outValues[writeIdx] = idx;
		writeIdx++;
	}
}

__global__ static void __launch_bounds__(MGS_DR_FIND_TILE_RANGES_WORKGROUP_SIZE)
_mgs_dr_forward_find_tile_ranges_kernel(uint32_t numRendered, const uint64_t* keys, uint2* outRanges)
{
	auto idx = cg::this_grid().thread_rank();
	if(idx >= numRendered)
		return;

	uint32_t tileIdx = (uint32_t)(keys[idx] >> 32);
	
	if(idx == 0)
		outRanges[tileIdx].x = 0;
	else
	{
		uint32_t prevTileIdx = (uint32_t)(keys[idx - 1] >> 32);
		if(tileIdx != prevTileIdx)
		{
			outRanges[prevTileIdx].y = idx;
			outRanges[tileIdx].x = idx;
		}
	}

	if(idx == numRendered - 1)
		outRanges[tileIdx].y = numRendered;
}

__global__ static void __launch_bounds__(MGS_DR_TILE_LEN)
_mgs_dr_forward_splat_kernel(MGSDRsettings settings, 
                             const uint2* ranges, const uint32_t* indices, const MGSDRgeomBuffers geom,
                             float* outColor, float* outAccumAlpha, uint32_t* outNumContributors)
{
	//compute pixel position:
	//---------------
	auto block = cg::this_thread_block();
	uint32_t tilesWidth = _mgs_ceildivide32(settings.width, MGS_DR_TILE_SIZE);

	uint32_t pixelMinX = block.group_index().x * MGS_DR_TILE_SIZE;
	uint32_t pixelMinY = block.group_index().y * MGS_DR_TILE_SIZE;

	uint32_t pixelMaxX = min(pixelMinX + MGS_DR_TILE_SIZE, settings.width );
	uint32_t pixelMaxY = min(pixelMinY + MGS_DR_TILE_SIZE, settings.height);

	uint32_t pixelX = pixelMinX + block.thread_index().x;
	uint32_t pixelY = pixelMinY + block.thread_index().y;
	
	uint32_t pixelId = pixelX + settings.width * pixelY;

	bool inside = pixelX < settings.width && pixelY < settings.height;

	//read gaussian range:
	//---------------
	uint2 range = ranges[block.group_index().x + tilesWidth * block.group_index().y];
	int32_t numToRender = range.y - range.x;
	uint32_t numRounds = _mgs_ceildivide32(numToRender, MGS_DR_TILE_LEN);

	//allocate shared memory:
	//---------------
	__shared__ QMvec2 collectedPixCenters  [MGS_DR_TILE_LEN];
	__shared__ QMvec4 collectedConicOpacity[MGS_DR_TILE_LEN];
	__shared__ QMvec3 collectedColor       [MGS_DR_TILE_LEN];

	//loop over batches until all threads are done:
	//---------------
	bool done = !inside;

	float accumAlpha = 1.0f;
	QMvec3 accumColor = qm_vec3_full(0.0f);

	uint32_t numContributors = 0;
	uint32_t lastContributor = 0;

	for(uint32_t i = 0; i < numRounds; i++)
	{
		//exit early if all threads done
		int numDone = __syncthreads_count(done);
		if(numDone == MGS_DR_TILE_LEN)
			break;

		//collectively load gaussian data
		uint32_t loadIdx = i * MGS_DR_TILE_LEN + block.thread_rank();
		if(range.x + loadIdx < range.y)
		{
			uint32_t gaussianIdx = indices[range.x + loadIdx];
			collectedPixCenters  [block.thread_rank()] = geom.pixCenters[gaussianIdx];
			collectedConicOpacity[block.thread_rank()] = geom.conicOpacity[gaussianIdx];
			collectedColor       [block.thread_rank()] = geom.color[gaussianIdx];
		}

		block.sync();

		//accumulate collected gaussians
		for(uint32_t j = 0; j < min(MGS_DR_TILE_LEN, numToRender); j++)
		{
			numContributors++;

			QMvec2 pos = collectedPixCenters[j];
			QMvec4 conicO = collectedConicOpacity[j];

			float dx = pos.x - (float)pixelX;
			float dy = pos.y - (float)pixelY;

			float power = -0.5f * (conicO.x * dx * dx + conicO.z * dy * dy) - conicO.y * dx * dy;
			if(power > 0.0f)
				continue;

			float alpha = min(MGS_DR_MAX_ALPHA, conicO.w * exp(power));
			if(alpha < MGS_DR_MIN_ALPHA)
				continue;
			
			float newAccumAlpha = accumAlpha * (1.0f - alpha);
			if(newAccumAlpha < MGS_DR_ACCUM_ALPHA_CUTOFF)
			{
				done = true;
				continue;
			}

			accumColor = qm_vec3_add(accumColor, qm_vec3_scale(collectedColor[j], alpha * accumAlpha));
			accumAlpha = newAccumAlpha;

			lastContributor = numContributors;
		}

		//decrement num left to render
		numToRender -= MGS_DR_TILE_LEN;
	}

	//write final color:
	//---------------
	if(inside)
	{
		outColor[pixelId * 3 + 0] = accumColor.r;
		outColor[pixelId * 3 + 1] = accumColor.g;
		outColor[pixelId * 3 + 2] = accumColor.b;

		outAccumAlpha[pixelId] = accumAlpha;
		outNumContributors[pixelId] = lastContributor;
	}
}

__device__ static void _mgs_dr_get_tile_bounds(uint32_t width, uint32_t height, QMvec2 pixCenter, float pixRadius, uint2& tileMin, uint2& tileMax)
{
	//TODO: use a smarter method to generate tiles
	//this generates far too many tiles for very anisotropic gaussians

	uint32_t tilesWidth  = _mgs_ceildivide32(width , MGS_DR_TILE_SIZE);
	uint32_t tilesHeight = _mgs_ceildivide32(height, MGS_DR_TILE_SIZE);
 
	tileMin.x = (uint32_t)min(max((int32_t)((pixCenter.x - pixRadius)                        / MGS_DR_TILE_SIZE), 0), tilesWidth );
	tileMin.y = (uint32_t)min(max((int32_t)((pixCenter.y - pixRadius)                        / MGS_DR_TILE_SIZE), 0), tilesHeight);
	tileMax.x = (uint32_t)min(max((int32_t)((pixCenter.x + pixRadius + MGS_DR_TILE_SIZE - 1) / MGS_DR_TILE_SIZE), 0), tilesWidth );
	tileMax.y = (uint32_t)min(max((int32_t)((pixCenter.y + pixRadius + MGS_DR_TILE_SIZE - 1) / MGS_DR_TILE_SIZE), 0), tilesHeight);
}