#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <tuple>

#include "cuda/mgs_dr_forward.h"
#include "cuda/mgs_dr_backward.h"

//-------------------------------------------//

std::function<uint8_t* (uint64_t size)> _mgs_dr_tensor_resize_function(at::Tensor& tensor)
{
    return [&tensor](uint64_t size) {
        tensor.resize_({(long long)size});
		return reinterpret_cast<uint8_t*>(tensor.contiguous().data_ptr());
    };
}

//-------------------------------------------//

//TODO: have this take in / return a struct, not a giant tuple
std::tuple<at::Tensor, int64_t, at::Tensor, at::Tensor, at::Tensor>
mgs_dr_forward(int64_t outWidth, int64_t outHeight, const at::Tensor& view, const at::Tensor& proj, double focalX, double focalY,
               const at::Tensor& means, const at::Tensor& scales, const at::Tensor& rotations, const at::Tensor& opacities, const at::Tensor& colors, const at::Tensor& harmonics,
               bool debug)
{
	//validate:
	//---------------
	uint32_t numGaussians = (uint32_t)means.size(0);

	//TODO

	//ensure outputs are contiguous

	//allocate output tensors:
	//---------------
	torch::TensorOptions floatOpts(torch::kFloat32);
	torch::TensorOptions byteOpts(torch::kByte);
	torch::Device device(torch::kCUDA);

	at::Tensor outImage = torch::full({ outHeight, outWidth, 3 }, 0.0f, floatOpts.device(device));

	at::Tensor geomBuf    = torch::empty({0}, byteOpts.device(device));
	at::Tensor binningBuf = torch::empty({0}, byteOpts.device(device));
	at::Tensor imageBuf   = torch::empty({0}, byteOpts.device(device));

	//render:
	//---------------	
	uint32_t numRendered = mgs_dr_forward_cuda(
		(uint32_t)outWidth, (uint32_t)outHeight,
		outImage.contiguous().data_ptr<float>(),

		view.contiguous().data_ptr<float>(),
		proj.contiguous().data_ptr<float>(),
		(float)focalX, (float)focalY,

		numGaussians,
		means    .contiguous().data_ptr<float>(),
		scales   .contiguous().data_ptr<float>(),
		rotations.contiguous().data_ptr<float>(),
		opacities.contiguous().data_ptr<float>(),
		colors   .contiguous().data_ptr<float>(),
		harmonics.contiguous().data_ptr<float>(),

		_mgs_dr_tensor_resize_function(geomBuf),
		_mgs_dr_tensor_resize_function(binningBuf),
		_mgs_dr_tensor_resize_function(imageBuf),

		debug
	);

	//return:
	//---------------
	return { 
		outImage,
		numRendered, geomBuf, binningBuf, imageBuf 
	};
}

//TODO: have this take in / return a struct, not a giant tuple
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mgs_dr_backward(const at::Tensor& dLdImage, const at::Tensor& view, const at::Tensor& proj, double focalX, double focalY,
                const at::Tensor& means, const at::Tensor& scales, const at::Tensor& rotations, const at::Tensor& opacities, const at::Tensor& colors, const at::Tensor& harmonics,
			    int64_t numRendered, const at::Tensor& geomBufs, const at::Tensor& binningBufs, const at::Tensor& imageBufs,
                bool debug)
{
	//validate:
	//---------------
	uint32_t numGaussians = (uint32_t)means.size(0);

	uint32_t width  = (uint32_t)dLdImage.size(1);
	uint32_t height = (uint32_t)dLdImage.size(0);

	//TODO

	//allocate output tensors:
	//---------------

	//TODO: ensure contiguous
	at::Tensor dLdMeans     = torch::zeros_like(means);
	at::Tensor dLdScales    = torch::zeros_like(scales);
	at::Tensor dLdRotations = torch::zeros_like(rotations);
	at::Tensor dLdOpacities = torch::zeros_like(opacities);
	at::Tensor dLdColors    = torch::zeros_like(colors);
	at::Tensor dLdHarmonics = torch::zeros_like(harmonics);

	//render:
	//---------------
	mgs_dr_backward_cuda(
		width, height, 
		dLdImage.contiguous().data_ptr<float>(),

		view.contiguous().data_ptr<float>(),
		proj.contiguous().data_ptr<float>(),
		(float)focalX, (float)focalY,

		numGaussians,
		means    .contiguous().data_ptr<float>(),
		scales   .contiguous().data_ptr<float>(),
		rotations.contiguous().data_ptr<float>(),
		opacities.contiguous().data_ptr<float>(),
		colors   .contiguous().data_ptr<float>(),
		harmonics.contiguous().data_ptr<float>(),

		numRendered,
		geomBufs   .data_ptr<uint8_t>(), //should already be contiguous
		binningBufs.data_ptr<uint8_t>(),
		imageBufs  .data_ptr<uint8_t>(),

		dLdMeans    .data_ptr<float>(),
		dLdScales   .data_ptr<float>(),
		dLdRotations.data_ptr<float>(),
		dLdOpacities.data_ptr<float>(),
		dLdColors   .data_ptr<float>(),
		dLdHarmonics.data_ptr<float>(),

		debug
	);

	//return:
	//---------------
	return {
		dLdMeans,
		dLdScales,
		dLdRotations,
		dLdOpacities,
		dLdColors,
		dLdHarmonics
	};
}

//-------------------------------------------//

//dummy module iniitalization
extern "C" {
	PyObject* PyInit__C(void)
	{
		static struct PyModuleDef module_def = {
			PyModuleDef_HEAD_INIT,
			"_C",
			NULL,
			-1,
			NULL,
		};
		return PyModule_Create(&module_def);
	}
}

TORCH_LIBRARY(mgs_diff_renderer, m) 
{
	m.def("forward(int outWidth, int outHeight, Tensor view, Tensor proj, float focalX, float focalY, Tensor means, Tensor scales, Tensor rotations, Tensor opacities, Tensor colors, Tensor harmonics, bool debug) -> (Tensor, int, Tensor, Tensor, Tensor)");
	m.def("backward(Tensor dLdImage, Tensor view, Tensor proj, float focalX, float focalY, Tensor means, Tensor scales, Tensor rotations, Tensor opacities, Tensor colors, Tensor harmonics, int numRendered, Tensor geomBufs, Tensor binningBufs, Tensor imageBufs, bool debug) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mgs_diff_renderer, CUDA, m) 
{
	m.impl("forward", &mgs_dr_forward);
	m.impl("backward", &mgs_dr_backward);
}