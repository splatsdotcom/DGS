#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

//-------------------------------------------//

extern "C" void mgs_diff_render_forward_cuda(const float* in, float* out, int64_t N);

//-------------------------------------------//

at::Tensor mgs_diff_render_forward(at::Tensor input) 
{
	//validate:
	//---------------
	TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
	TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

	if(!input.is_contiguous())
		input = input.contiguous();

	//run cuda:
	//---------------
	at::Tensor output = torch::empty_like(input);
	int64_t size = input.numel();

	mgs_diff_render_forward_cuda(input.data_ptr<float>(), output.data_ptr<float>(), size);
	return output;
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
	m.def("forward(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(mgs_diff_renderer, CUDA, m) 
{
	m.impl("forward", &mgs_diff_render_forward);
}