/* mgs_dr_global.h
 *
 * contains utility functions, macros, and global config
 * for the differentiable renderer
 */

#ifndef MGS_DR_GLOBAL_H
#define MGS_DR_GLOBAL_H

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <chrono>

// only enable for personal testing:
// #define MGS_DR_PROFILE

//-------------------------------------------//

#define MGS_DR_CUDA_ERROR_CHECK(s)                                        \
	s;                                                                    \
	if(debug)                                                             \
	{                                                                     \
		cudaError error = cudaDeviceSynchronize();                        \
		if(error != cudaSuccess)                                          \
		{                                                                 \
			std::cerr << std::endl << "MGSDR: CUDA error in \"" <<        \
				__FILENAME__ << "\" at line " << __LINE__ <<              \
				": \"" << cudaGetErrorString(error) << "\"" << std::endl; \
			throw std::runtime_error("MGSDR CUDA error");                 \
		}                                                                 \
	}

#ifdef MGS_DR_PROFILE
	#define MGS_DR_PROFILE_REGION_START(name) cudaDeviceSynchronize(); auto tStart##name = std::chrono::high_resolution_clock::now()
	#define MGS_DR_PROFILE_REGION_END(name) cudaDeviceSynchronize(); auto tEnd##name = std::chrono::high_resolution_clock::now()

	#define MGS_DR_PROFILE_REGION_TIME(name) std::chrono::duration_cast<std::chrono::microseconds>(tEnd##name - tStart##name).count() / 1000.0
#else
	#define MGS_DR_PROFILE_REGION_START(name)
	#define MGS_DR_PROFILE_REGION_END(name)

	#define MGS_DR_PROFILE_REGION_TIME(name) 0.0
#endif

#endif //#ifndef MGS_DR_GLOBAL_H
