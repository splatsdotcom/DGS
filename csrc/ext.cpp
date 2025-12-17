#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <optional>
#include <string>

#include "dgs_gaussians.h"
#include "dgs_encode.h"
#include "dgs_decode.h"

namespace py = pybind11;

//-------------------------------------------//

struct DGSgaussiansDeleter
{
	void operator()(DGSgaussians* gaussians)
	{
		dgs_gaussians_free(gaussians);
	}
};

struct DGSgaussiansFdeleter
{
	void operator()(DGSgaussiansF* gaussians)
	{
		dgs_gaussiansf_free(gaussians);
	}
};

std::shared_ptr<DGSgaussians> gaussians_make_shared()
{
	return std::shared_ptr<DGSgaussians>(new DGSgaussians, DGSgaussiansDeleter{});
}

std::shared_ptr<DGSgaussiansF> gaussiansf_make_shared()
{
	return std::shared_ptr<DGSgaussiansF>(new DGSgaussiansF, DGSgaussiansFdeleter{});
}

//-------------------------------------------//

std::shared_ptr<DGSgaussians> read_or_decode(const py::object obj, DGSmetadata& metadata);

//-------------------------------------------//

PYBIND11_MODULE(_C, m)
{
	m.doc() = "DGS Core";

	constexpr uint32_t DGS_VERSION_MAJOR = (DGS_VERSION >> 22) & 0x3FF;
	constexpr uint32_t DGS_VERSION_MINOR = (DGS_VERSION >> 12) & 0x3FF;
	constexpr uint32_t DGS_VERSION_PATCH = DGS_VERSION & 0x3FF;

	m.attr("__version__") =
		std::to_string(DGS_VERSION_MAJOR) + "." +
		std::to_string(DGS_VERSION_MINOR) + "." +
		std::to_string(DGS_VERSION_PATCH);

	py::class_<DGSgaussians, std::shared_ptr<DGSgaussians>>(m, "Gaussians")
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
			if(shDegree > DGS_GAUSSIANS_MAX_SH_DEGREE)
				throw std::invalid_argument("harmonics degree is too large");

			//load into DGSgaussiansF:
			//---------------
			DGSgaussiansF gaussians;
			gaussians.count      = (uint32_t)count;
			gaussians.shDegree   = shDegree;
			gaussians.dynamic    = (dgs_bool_t)dynamic;
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
			std::shared_ptr<DGSgaussians> out = gaussians_make_shared();

			DGSerror error = dgs_gaussians_from_fp32(&gaussians, out.get());
			if(error != DGS_SUCCESS)
				throw std::runtime_error("DGS internal error: \"" + std::string(dgs_error_get_description(error)) + "\"");

			return out;
		}),
		"Load Gaussians from NumPy/PyTorch arrays.",
		py::arg("means"),
		py::arg("scales"),
		py::arg("rotations"),
		py::arg("opacities"),
		py::arg("shs"),
		py::arg("velocities") = py::none(),
		py::arg("t_means")     = py::none(),
		py::arg("t_stdevs")    = py::none())

		.def("__len__", [](const DGSgaussians& self)
		{
			return self.count;
		});

	py::class_<DGSmetadata>(m, "Metadata")
		.def(py::init([](float duration)
		{
			//validate:
			//---------------
			if(duration < 0.0f)
				throw std::invalid_argument("duration must be positive");

			//create struct:
			//---------------
			DGSmetadata metadata;
			metadata.duration = duration;

			return metadata;
		}),
		"Initialize .dgs metadata",
		py::arg("duration") = 0.0f)

		.def_readwrite("duration", &DGSmetadata::duration);

	m.def("encode", [](const DGSgaussians& gaussians, const DGSmetadata& metadata, const std::string& path)
	{
		DGSerror error = dgs_encode(&gaussians, metadata, path.c_str());
		if(error != DGS_SUCCESS)
			throw std::runtime_error("DGS internal error: \"" + std::string(dgs_error_get_description(error)) + "\"");
	},
	"Encode a set of Gaussians into a .dgs file.",
	py::arg("gaussians"),
	py::arg("metadata"),
	py::arg("out_path"));

	m.def("decode", [](const std::string& path)
	{
		std::shared_ptr<DGSgaussians> out = gaussians_make_shared();
		DGSmetadata outMetadata;

		DGSerror error = dgs_decode_from_file(path.c_str(), out.get(), &outMetadata);
		if(error != DGS_SUCCESS)
			throw std::runtime_error("DGS internal error: \"" + std::string(dgs_error_get_description(error)) + "\"");

		return std::make_tuple(out, outMetadata);
	},
	"Decode a set of Gaussians from a .dgs file.\n",
	"Returns a tuple (Gaussians, Metadata).",
	py::arg("path"));

	m.def("combine", [](py::object g1Obj, py::object g2Obj, py::object outPathObj)
	{
		//load:
		//---------------
		DGSmetadata g1Meta, g2Meta;

		std::shared_ptr<DGSgaussians> g1 = read_or_decode(g1Obj, g1Meta);
		std::shared_ptr<DGSgaussians> g2 = read_or_decode(g2Obj, g2Meta);

		//combine:
		//---------------
		std::shared_ptr<DGSgaussians> out = gaussians_make_shared();

		DGSerror error = dgs_gaussians_combine(g1.get(), g2.get(), out.get());
		if(error != DGS_SUCCESS)
			throw std::runtime_error("DGS internal error: \"" + std::string(dgs_error_get_description(error)) + "\"");

		//write out if path provided:
		//---------------
		if(!outPathObj.is_none())
		{
			std::string outPath = outPathObj.cast<std::string>();

			error = dgs_encode(out.get(), g1Meta, outPath.c_str());
			if(error != DGS_SUCCESS)
				throw std::runtime_error("DGS internal error: \"" + std::string(dgs_error_get_description(error)) + "\"");
		}

		//return:
		//---------------
		return out;
	},
	"Combine two Gaussian sets.\n"
	"Arguments may be either Gaussians objects or file paths.\n"
	"If out_path is supplied, the result is written to disk.",
	py::arg("g1"),
	py::arg("g2"),
	py::arg("out_path") = py::none());
}

//-------------------------------------------//

std::shared_ptr<DGSgaussians> read_or_decode(const py::object obj, DGSmetadata& metadata)
{
	std::shared_ptr<DGSgaussians> out;

	if(py::isinstance<py::str>(obj))
	{
		std::string path = obj.cast<std::string>();
		out = gaussians_make_shared();

		DGSerror error = dgs_decode_from_file(path.c_str(), out.get(), &metadata);
		if(error != DGS_SUCCESS)
			throw std::runtime_error("DGS internal error: \"" + std::string(dgs_error_get_description(error)) + "\"");
	}
	else
	{
		try 
		{
			out = obj.cast<std::shared_ptr<DGSgaussians>>();
			metadata = {0};
		}
		catch(...) 
		{
			throw std::invalid_argument("Argument must be either str or Gaussians");
		}
	}

	return out;
}