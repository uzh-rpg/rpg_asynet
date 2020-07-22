#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <conv2d.h>
#include <rulebook.h>

namespace py = pybind11;

PYBIND11_MODULE(async_sparse, m) {
    // optional module docstring
    m.doc() = "pybind11 example plugin";

    py::class_<AsynSparseConvolution2D>(m, "AsynSparseConvolution2D")
        .def(py::init<int,int,int,int,bool,bool, bool>())
        .def("forward", &AsynSparseConvolution2D::forward, py::return_value_policy::reference_internal)
        .def("setParameters", &AsynSparseConvolution2D::setParameters)
        .def("initMaps", &AsynSparseConvolution2D::initMaps)
        .def("initActiveMap", &AsynSparseConvolution2D::initActiveMap, py::return_value_policy::reference_internal);

    py::class_<RuleBook>(m, "RuleBook")
        .def(py::init<int,int,int,int>())
        .def("print", &RuleBook::print);
}
