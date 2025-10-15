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

#endif