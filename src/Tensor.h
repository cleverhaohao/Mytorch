#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstddef>
#include <cublas_v2.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class Tensor {
public:
    // 构造函数
    Tensor(std::vector<int> shape, bool device = false, std::vector<float> data = std::vector<float>(0), std::vector<float> grad = std::vector<float>(0));
    Tensor(py::array_t<float> input_array);
    // 析构函数
    ~Tensor();

    // 重载运算符用于索引访问
    // float& operator()(std::vector<int>& indices);

    // float& grad_operator(std::vector<int> indices);

    std::vector<float> get_data();
    std::vector<float> get_grad();
    py::array_t<float> data();
    py::array_t<float> grad();  

    // Tensor& cpu();
    // Tensor& gpu();

    // void relu_cpu();
    void relu_gpu();
    // void sigmoid_cpu();
    void sigmoid_gpu();
    void relu_backward_gpu(float* gradOutput);
    void sigmoid_backward_gpu(float* gradOutput);



    // 成员变量
    std::vector<int> shape;      // 张量的形状                
    bool onGPU;                  // 标记张量是否在GPU上
    float* deviceData;           // GPU上的数据指针
    float* deviceGrad;           // GPU上的梯度指针
    size_t size;                // 张量的总大小
    float* maxpooling_mask;
};

class Fully_Connected{
public:

    Fully_Connected(int input_size, int output_size, bool onGPU = false, std::vector<float> data = std::vector<float>(0));
    Fully_Connected(py::array_t<float> input_array);
    void forward(Tensor& input, Tensor& output);
    void backward(Tensor& input, Tensor& output);
    Tensor weight;
    cublasHandle_t handle;
    int input_size;
    int output_size;
    bool onGPU;
};

class Convolutional{
public:
    Convolutional(int in_channels, int out_channels, int kernel_size, int padding, bool onGPU = false, std::vector<float> data = std::vector<float>(0));

    void forward(Tensor& input, Tensor& output);
    void backward(Tensor& input, Tensor& output);
    Tensor weight;
    cublasHandle_t handle;
    int in_channels;
    int out_channels;
    int kernel_size;
    int padding;
    bool onGPU;
};

class MaxPooling {
public:
    MaxPooling(int kernel_size, bool onGPU = false);

    void forward(Tensor& input, Tensor& output);
    void backward(Tensor& input, Tensor& output);

    int kernel_size;
    bool onGPU;
};


py::array_t<float> Softmax(Tensor& input);

py::array_t<float> CrossEntropyLoss(Tensor& softmax_output, Tensor& labels);

py::array_t<float> CrossEntropyLoss_backward_with_Softmax(Tensor& softmax_output, Tensor& labels);

py::array_t<float> fc_forward(Tensor& input, Tensor& weight);

std::tuple<py::array_t<float>, py::array_t<float>> fc_backward(Tensor& input, Tensor& weight, Tensor& gradOutput);

py::array_t<float> conv_forward(Tensor& input, Tensor& weight);

std::tuple<py::array_t<float>, py::array_t<float>> conv_backward(Tensor& input, Tensor& weight, Tensor& gradOutput);

std::tuple<py::array_t<float>, py::array_t<float>> maxpooling_forward(Tensor& input);

py::array_t<float> maxpooling_backward(Tensor& input, Tensor& gradOutput, Tensor& mask);

#endif // TENSOR_H
