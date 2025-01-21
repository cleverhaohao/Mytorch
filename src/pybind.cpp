#include <vector>
#include "Tensor.h"
#include<pybind11/pybind11.h>
#include<pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
PYBIND11_MODULE(mytensor, m) {
py::class_<Tensor>(m,"Tensor")
.def(py::init<std::vector<int>, bool, std::vector<float>, std::vector<float>>())
.def(py::init<py::array_t<float>>(), "Constructor from numpy array")
.def("get_data", &Tensor::get_data)
.def("get_grad", &Tensor::get_grad)
.def("data", &Tensor::data)
.def("grad", &Tensor::grad)
// .def("cpu", &Tensor::cpu)
// .def("gpu", &Tensor::gpu)
//.def("relu_cpu", &Tensor::relu_cpu)
.def("relu_gpu", &Tensor::relu_gpu)
//.def("sigmoid_cpu", &Tensor::sigmoid_cpu)
.def("sigmoid_gpu", &Tensor::sigmoid_gpu)
.def("relu_backward_gpu", &Tensor::relu_backward_gpu)
.def("sigmoid_backward_gpu", &Tensor::sigmoid_backward_gpu)
//.def("__call__", &Tensor::operator(), py::arg("indices"))
//.def("grad_operator", &Tensor::grad_operator, py::arg("indices"))
.def_readwrite("onGPU", &Tensor::onGPU)
.def_readwrite("shape", &Tensor::shape)
.def_readwrite("size", &Tensor::size);

py::class_<Fully_Connected>(m,"Fully_Connected")
.def(py::init<int, int, bool, std::vector<float>>())
.def(py::init<py::array_t<float>>(), "Constructor from numpy array")
.def("forward", &Fully_Connected::forward)
.def("backward", &Fully_Connected::backward)
.def_readwrite("weight", &Fully_Connected::weight)
.def_readwrite("input_size", &Fully_Connected::input_size)
.def_readwrite("output_size", &Fully_Connected::output_size)
.def_readwrite("onGPU", &Fully_Connected::onGPU);


py::class_<Convolutional>(m,"Convolutional")
.def(py::init<int, int, int, int, bool, std::vector<float>>())
.def("forward", &Convolutional::forward)
.def("backward", &Convolutional::backward)
.def_readwrite("weight", &Convolutional::weight);


py::class_<MaxPooling>(m,"MaxPooling")
.def(py::init<int, bool>())
.def("forward", &MaxPooling::forward)
.def("backward", &MaxPooling::backward)
.def_readwrite("kernel_size", &MaxPooling::kernel_size)
.def_readwrite("onGPU", &MaxPooling::onGPU);

m.def("Softmax", &Softmax, "Compute Softmax for a given tensor");

m.def("CrossEntropyLoss", &CrossEntropyLoss, "Compute CrossEntropy for a given tensor");

m.def("CrossEntropyLoss_backward_with_Softmax", &CrossEntropyLoss_backward_with_Softmax, "Compute CrossEntropy for a given tensor");

m.def("fc_forward", &fc_forward, "Compute forward result given input and fc_weight");
m.def("fc_backward", &fc_backward, "Compute fullyconnected layer backward result ");

m.def("conv_forward", &conv_forward, "Compute forward result given input and conv_kernel");
m.def("conv_backward", &conv_backward, "Compute conv layer backward result ");

m.def("maxpooling_forward", &maxpooling_forward, "Compute maxpooling_forward result given input");
m.def("maxpooling_backward", &maxpooling_backward, "Compute maxpooling backward result");
}