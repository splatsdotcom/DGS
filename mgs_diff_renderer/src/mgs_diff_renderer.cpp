#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include "cuda/mgs_dr_forward.h"

//-------------------------------------------//

std::function<uint8_t* (uint64_t size)> _mgs_dr_tensor_resize_function(at::Tensor& tensor)
{
    return [&tensor](uint64_t size) {
        tensor.resize_({(long long)size});
		return reinterpret_cast<uint8_t*>(tensor.contiguous().data_ptr());
    };
}

//-------------------------------------------//

at::Tensor mgs_dr_forward(int64_t outWidth, int64_t outHeight, const at::Tensor& view, const at::Tensor& proj, double focalX, double focalY,
                          const at::Tensor& means, const at::Tensor& scales, const at::Tensor& rotations, const at::Tensor& opacities, const at::Tensor& colors, const at::Tensor& harmonics)
{
	//validate:
	//---------------

	//TODO

	//allocate output tensors:
	//---------------
	torch::TensorOptions floatOpts(torch::kFloat32);
	torch::TensorOptions byteOpts(torch::kByte);
	torch::Device device(torch::kCUDA);

	at::Tensor outImage = torch::full({ outHeight, outWidth, 4 }, 0.0f, floatOpts.device(device));

	at::Tensor geomBuf    = torch::empty({0}, byteOpts.device(device));
	at::Tensor binningBuf = torch::empty({0}, byteOpts.device(device));
	at::Tensor imageBuf   = torch::empty({0}, byteOpts.device(device));

	//render:
	//---------------
	uint32_t numGaussians = (uint32_t)means.size(0);
	
	mgs_dr_forward_cuda(
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
		_mgs_dr_tensor_resize_function(imageBuf)
	);

	//return:
	//---------------
	return outImage;
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
	m.def("forward(int outWidth, int outHeight, Tensor view, Tensor proj, float focalX, float focalY, Tensor means, Tensor scales, Tensor rotations, Tensor opacities, Tensor colors, Tensor harmonics) -> Tensor");
}

TORCH_LIBRARY_IMPL(mgs_diff_renderer, CUDA, m) 
{
	m.impl("forward", &mgs_dr_forward);
}