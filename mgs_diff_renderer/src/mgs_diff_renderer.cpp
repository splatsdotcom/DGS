#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <torch/custom_class.h>
#include <tuple>

#include "cuda/mgs_dr_forward.h"
#include "cuda/mgs_dr_backward.h"

//-------------------------------------------//

MGSDRresizeFunc _mgs_dr_tensor_resize_function(at::Tensor& tensor)
{
    return [&tensor](uint64_t size) {
        tensor.resize_({(long long)size});
		return reinterpret_cast<uint8_t*>(tensor.contiguous().data_ptr());
    };
}

//-------------------------------------------//

class MGSDRsettingsTorch : public torch::CustomClassHolder
{
public:
	MGSDRsettingsTorch(int64_t width, int64_t height, const at::Tensor& view, const at::Tensor& proj,
	                   double focalX, double focalY, bool debug)
	{
		//validate:
		//---------------
		if(width <= 0 || height <= 0)
			throw std::invalid_argument("Image dimensions must be positive");
		if(width > UINT32_MAX || height > UINT32_MAX || width * height > UINT32_MAX)
			throw std::invalid_argument("Image dimensions are too large! Must be < UINT32_MAX pixels");

		if(view.dtype() != torch::kFloat32 || proj.dtype() != torch::kFloat32)
			throw std::invalid_argument("View and projection matrices must be float32");
		if(view.dim() != 2 || view.size(0) != 4 || view.size(1) != 4 ||
		   proj.dim() != 2 || proj.size(0) != 4 || proj.size(1) != 4)
			throw std::invalid_argument("View and projection matrices must have shape (4, 4)");

		if(focalX <= 0.0f || focalY <= 0.0f)
			throw std::invalid_argument("Focal lengths must be positive");

		//initialize:
		//---------------
		settings.width = (uint32_t)width;
		settings.height = (uint32_t)height;
		settings.view = qm_mat4_load_row_major(view.to(torch::kCPU).contiguous().data_ptr<float>());
		settings.proj = qm_mat4_load_row_major(proj.to(torch::kCPU).contiguous().data_ptr<float>());
		settings.viewProj = qm_mat4_mult(settings.proj, settings.view);
		settings.focalX = (float)focalX;
		settings.focalY = (float)focalY;
		settings.debug = debug;
	}

	MGSDRsettings settings;
};

//-------------------------------------------//

std::tuple<at::Tensor, int64_t, at::Tensor, at::Tensor, at::Tensor>
mgs_dr_forward(const c10::intrusive_ptr<MGSDRsettingsTorch>& settings,
               const at::Tensor& means, const at::Tensor& scales, const at::Tensor& rotations, const at::Tensor& opacities, const at::Tensor& harmonics)
{
	MGSDRsettings cSettings = settings->settings;

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

	at::Tensor outImage = torch::full({ cSettings.height, cSettings.width, 3 }, 0.0f, floatOpts.device(device));

	at::Tensor geomBuf    = torch::empty({0}, byteOpts.device(device));
	at::Tensor binningBuf = torch::empty({0}, byteOpts.device(device));
	at::Tensor imageBuf   = torch::empty({0}, byteOpts.device(device));

	//populate gaussians struct:
	//---------------
	MGSDRgaussians gaussians;
	gaussians.count = numGaussians;
	gaussians.means     = means    .contiguous().data_ptr<float>();
	gaussians.scales    = scales   .contiguous().data_ptr<float>();
	gaussians.rotations = rotations.contiguous().data_ptr<float>();
	gaussians.opacities = opacities.contiguous().data_ptr<float>();
	gaussians.harmonics = harmonics.contiguous().data_ptr<float>();

	//render:
	//---------------	
	uint32_t numRendered = mgs_dr_forward_cuda(
		cSettings, gaussians,
		
		_mgs_dr_tensor_resize_function(geomBuf),
		_mgs_dr_tensor_resize_function(binningBuf),
		_mgs_dr_tensor_resize_function(imageBuf),
		
		outImage.contiguous().data_ptr<float>()
	);

	//return:
	//---------------
	return { 
		outImage,
		numRendered, geomBuf, binningBuf, imageBuf 
	};
}

//TODO: have this take in / return a struct, not a giant tuple
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mgs_dr_backward(const c10::intrusive_ptr<MGSDRsettingsTorch>& settings, const at::Tensor& dLdImage,
                const at::Tensor& means, const at::Tensor& scales, const at::Tensor& rotations, const at::Tensor& opacities, const at::Tensor& harmonics,
			    int64_t numRendered, const at::Tensor& geomBufs, const at::Tensor& binningBufs, const at::Tensor& imageBufs)
{
	MGSDRsettings cSettings = settings->settings;

	//validate:
	//---------------
	uint32_t numGaussians = (uint32_t)means.size(0);

	uint32_t width  = (uint32_t)dLdImage.size(1);
	uint32_t height = (uint32_t)dLdImage.size(0);

	//TODO

	//allocate output tensors:
	//---------------
	at::Tensor dLdMeans     = torch::zeros_like(means)    .contiguous();
	at::Tensor dLdScales    = torch::zeros_like(scales)   .contiguous();
	at::Tensor dLdRotations = torch::zeros_like(rotations).contiguous();
	at::Tensor dLdOpacities = torch::zeros_like(opacities).contiguous();
	at::Tensor dLdHarmonics = torch::zeros_like(harmonics).contiguous();

	MGSDRgaussians dLdGaussians;
	dLdGaussians.means     = dLdMeans    .data_ptr<float>();
	dLdGaussians.scales    = dLdScales   .data_ptr<float>();
	dLdGaussians.rotations = dLdRotations.data_ptr<float>();
	dLdGaussians.opacities = dLdOpacities.data_ptr<float>();
	dLdGaussians.harmonics = dLdHarmonics.data_ptr<float>();

	//populate gaussians struct:
	//---------------
	MGSDRgaussians gaussians;
	gaussians.count = numGaussians;
	gaussians.means     = means    .contiguous().data_ptr<float>();
	gaussians.scales    = scales   .contiguous().data_ptr<float>();
	gaussians.rotations = rotations.contiguous().data_ptr<float>();
	gaussians.opacities = opacities.contiguous().data_ptr<float>();
	gaussians.harmonics = harmonics.contiguous().data_ptr<float>();

	//render:
	//---------------
	mgs_dr_backward_cuda(
		cSettings, dLdImage.contiguous().data_ptr<float>(), gaussians,

		numRendered,
		geomBufs   .data_ptr<uint8_t>(), //should already be contiguous
		binningBufs.data_ptr<uint8_t>(),
		imageBufs  .data_ptr<uint8_t>(),

		dLdGaussians
	);

	//return:
	//---------------
	return {
		dLdMeans,
		dLdScales,
		dLdRotations,
		dLdOpacities,
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
	m.class_<MGSDRsettingsTorch>("Settings")
		.def(torch::init<int64_t, int64_t, const at::Tensor&, const at::Tensor&, double, double, bool>());

	m.def(
		"forward(__torch__.torch.classes.mgs_diff_renderer.Settings settings, Tensor means, Tensor scales, Tensor rotations, Tensor opacities, Tensor harmonics) -> (Tensor, int, Tensor, Tensor, Tensor)",
		&mgs_dr_forward
	);
	m.def(
		"backward(__torch__.torch.classes.mgs_diff_renderer.Settings settings, Tensor dLdImage, Tensor means, Tensor scales, Tensor rotations, Tensor opacities, Tensor harmonics, int numRendered, Tensor geomBufs, Tensor binningBufs, Tensor imageBufs) -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
		&mgs_dr_backward
	);
}
