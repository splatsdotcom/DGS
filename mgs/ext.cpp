#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <optional>
#include <string>

#include "mgs_gaussians.h"
#include "mgs_encode.h"

namespace py = pybind11;

//-------------------------------------------//

PYBIND11_MODULE(_C, m)
{
	m.doc() = "MGS Core";

	py::class_<MGSgaussians, std::shared_ptr<MGSgaussians>>(m, "Gaussians")
		.def(py::init([](const py::array_t<float, py::array::c_style | py::array::forcecast>& means,
		                 const py::array_t<float, py::array::c_style | py::array::forcecast>& scales,
		                 const py::array_t<float, py::array::c_style | py::array::forcecast>& rotations,
		                 const py::array_t<float, py::array::c_style | py::array::forcecast>& opacities,
		                 const py::array_t<float, py::array::c_style | py::array::forcecast>& shs,
		                 py::object velocitiesObj,
		                 py::object tMeansObj,
		                 py::object tStdevsObj)
		{
			//determine if dynamic:
			//---------------
			bool dynamic = !velocitiesObj.is_none() && !tMeansObj.is_none() && !tStdevsObj.is_none();
			if(!dynamic && (!velocitiesObj.is_none() || !tMeansObj.is_none() || !tStdevsObj.is_none()))
				throw std::invalid_argument("all of { velocities, tMeans, tStdevs } must be non-None for dynamic gaussians");

			py::array_t<float> velocities;
			py::array_t<float> tMeans;
			py::array_t<float> tStdevs;
			if(dynamic)
			{
				velocities = velocitiesObj.cast<py::array_t<float, py::array::c_style>>();
				tMeans = tMeansObj.cast<py::array_t<float, py::array::c_style>>();
				tStdevs = tStdevsObj.cast<py::array_t<float, py::array::c_style>>();
			}

			//validate:
			//---------------
			int64_t count = means.shape(0);

			if(means.ndim() != 2 || means.shape(0) != count || means.shape(1) != 3)
				throw std::invalid_argument("means must have shape (N, 3)");
			if(scales.ndim() != 2 || scales.shape(0) != count || scales.shape(1) != 3)
				throw std::invalid_argument("scales must have shape (N, 3)");
			if(rotations.ndim() != 2 || rotations.shape(0) != count || rotations.shape(1) != 4)
				throw std::invalid_argument("rotations must have shape (N, 4)");
			if(opacities.ndim() != 2 || opacities.shape(0) != count || opacities.shape(1) != 1)
				throw std::invalid_argument("opacities must have shape (N, 1)");
			if(shs.ndim() != 3 || shs.shape(0) != count || shs.shape(2) != 3)
				throw std::invalid_argument("harmonics must have shape (N, (degree+1)^2, 3)");

			if(dynamic)
			{
				if(velocities.ndim() != 2 || velocities.shape(0) != count || velocities.shape(1) != 3)
					throw std::invalid_argument("velocities must have shape (N, 3)");
				if(tMeans.ndim() != 2 || tMeans.shape(0) != count || tMeans.shape(1) != 1)
					throw std::invalid_argument("tMeans must have shape (N, 1)");
				if(tStdevs.ndim() != 2 || tStdevs.shape(0) != count || tStdevs.shape(1) != 1)
					throw std::invalid_argument("tStdevs must have shape (N, 1)");
			}

			uint32_t shDegree = 0;
			while((int64_t)(shDegree + 1) * (shDegree + 1) < shs.shape(1))
				shDegree++;

			if((int64_t)(shDegree + 1) * (shDegree + 1) != shs.shape(1))
				throw std::invalid_argument("harmonics have an invalid degree");
			if(shDegree > MGS_GAUSSIANS_MAX_SH_DEGREE)
				throw std::invalid_argument("harmonics degree is too large");

			//load into MGSgaussiansF:
			//---------------
			MGSgaussiansF gaussians;
			gaussians.count = (uint32_t)count;
			gaussians.shDegree = shDegree;
			gaussians.dynamic = (mgs_bool_t)dynamic;
			gaussians.means      = (QMvec3*)means.data();
			gaussians.scales     = (QMvec3*)scales.data();
			gaussians.rotations  = (QMquaternion*)rotations.data();
			gaussians.opacities  = (float*)opacities.data();
			gaussians.shs        = (float*)shs.data();

			gaussians.velocities = dynamic ? (QMvec3*)velocities.data() : nullptr;
			gaussians.tMeans     = dynamic ? (float*)tMeans.data()      : nullptr;
			gaussians.tStdevs    = dynamic ? (float*)tStdevs.data()     : nullptr;

			//return:
			//---------------
			std::shared_ptr<MGSgaussians> out = std::make_shared<MGSgaussians>();

			MGSerror error = mgs_gaussians_from_fp32(&gaussians, out.get());
			if(error != MGS_SUCCESS)
				throw std::runtime_error("MGS internal error: \"" + std::string(mgs_error_get_description(error)) + "\"");

			return out;
		}),
		"Load Gaussians from NumPy/PyTorch arrays",
		py::arg("means"),
		py::arg("scales"),
		py::arg("rotations"),
		py::arg("opacities"),
		py::arg("shs"),
		py::arg("velocities") = py::none(),
		py::arg("tMeans")     = py::none(),
		py::arg("tStdevs")    = py::none())

		.def("__len__", [](const MGSgaussians& self)
		{
			return self.count;
		});

	m.def("encode", [](const MGSgaussians& gaussians, const std::string& path)
	{
		MGSerror error = mgs_encode(&gaussians, path.c_str());
		if(error != MGS_SUCCESS)
			throw std::runtime_error("MGS internal error: \"" + std::string(mgs_error_get_description(error)) + "\"");
	});
}