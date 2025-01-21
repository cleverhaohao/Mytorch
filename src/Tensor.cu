#include <iostream>
#include <vector>
#include "Tensor.h" 
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
using namespace std;

__global__ void reluKernel(float* data, size_t size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            data[idx] = max(0.0f, data[idx]);
        }
    }

__global__ void sigmoidKernel(float* data, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f / (1.0f + exp(-data[idx]));
    }
}

__global__ void relu_backwardKernel(float* data, float* gradOutput, float* gradInput, size_t size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            if (data[idx] > 0) {
                gradInput[idx] = gradOutput[idx]; 
            } else {
                gradInput[idx] = 0; 
            }
        }
    }

__global__ void sigmoid_backwardKernel(float* data, float* gradOutput, float* gradInput, size_t size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            gradInput[idx] = gradOutput[idx] * data[idx] * (1 - data[idx]);
        }
        }
__global__ void float_to_int_kernel(const float* float_labels, int* int_labels, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int_labels[idx] = static_cast<int>(float_labels[idx]);
    }
}

void gemm_gpu(cublasOperation_t transa, cublasOperation_t transb, const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda = m, ldb = k, ldc = m;
    const float alf = 1, bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // 创建 cuBLAS 句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 执行矩阵乘法
    if (transa == CUBLAS_OP_N && transb == CUBLAS_OP_N){
        cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);}
        
    else if (transa == CUBLAS_OP_T && transb == CUBLAS_OP_N){
        cublasSgemm(handle, transa, transb, m, n, k, alpha, A, k, B, ldb, beta, C, ldc);}
        
    else if (transa == CUBLAS_OP_N && transb == CUBLAS_OP_T){
        cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, n, beta, C, ldc);}
        
    else if (transa == CUBLAS_OP_T && transb == CUBLAS_OP_T){
        cublasSgemm(handle, transa, transb, m, n, k, alpha, A, k, B, n, beta, C, ldc);}

    // 销毁 cuBLAS 句柄
    cublasDestroy(handle);
}


__global__ void im2col_kernel(const float *im, float *col, const int Batchsize, const int C_in, const int H, const int W, const int K, const int pad) {
   
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= Batchsize * H * W * C_in * K * K) {
        return;}
    int index_col = index;
    int k_w = index % K;
    index /= K;
    int k_h = index % K;
    index /= K;
    int c_in = index % C_in;
    index /= C_in;
    int w_out = index % W;
    index /= W;
    int h_out = index % H;
    index /= H;
    int n = index;

    int index_im = n * C_in * H * W + c_in * H * W + (h_out + k_h - pad) * W + (w_out + k_w - pad);
    col[index_col] = (h_out + k_h - pad) >= 0 && (h_out + k_h - pad) < H && (w_out + k_w - pad) >= 0 && (w_out + k_w - pad) < W ? im[index_im] : 0;

}

void im2col_gpu(const float *im, float *col, const int Batchsize, const int C_in, const int H, const int W, const int K, const int pad) {
    int size = Batchsize * H * W * C_in * K * K;
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    im2col_kernel<<<grid_size, block_size>>>(im, col, Batchsize, C_in, H, W, K, pad);
}


__global__ void change_shape_kernel(const float *input, float *output, const int N, const int C_out, const int H, const int W) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C_out * H * W) {
        return;
    }
    int index_output = index;
    int w = index % W;
    index /= W;
    int h = index % H;
    index /= H;
    int c_out = index % C_out;
    index /= C_out;
    int n = index;

    output[index_output] = input[n * C_out * H * W + h * W * C_out + w * C_out + c_out];
}

void change_shape_gpu(const float *input, float *output, const int N, const int C_out, const int H, const int W) {
    int size = N * C_out * H * W;
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    change_shape_kernel<<<grid_size, block_size>>>(input, output, N, C_out, H, W);
}

__global__ void reverse_change_shape_kernel(const float *input, float *output, const int N, const int C_out, const int H, const int W) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C_out * H * W) {
        return;
    }
    int index_output = index;
    int c_out = index % C_out;
    index /= C_out;
    int w = index % W;
    index /= W;
    int h = index % H;
    index /= H;
    int n = index;

    output[index_output] = input[n * H * W * C_out + c_out * H * W + h * W + w];
}

void reverse_change_shape_gpu(const float *input, float *output, const int N, const int C_out, const int H, const int W) {
    int size = N * C_out * H * W;
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    reverse_change_shape_kernel<<<grid_size, block_size>>>(input, output, N, C_out, H, W);
}

__global__ void col2im_kernel(const float *col, float *im, const int Batchsize, const int C_in, const int H, const int W, const int K, const int pad) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= Batchsize * H * W * C_in) {
        return;
    }
    int index_im = index;
    int w = index % W;
    index /= W;
    int h = index % H;
    index /= H;
    int c_in = index % C_in;
    index /= C_in;
    int n = index;

    float value = 0;
    for (int k_h = 0; k_h < K; ++k_h) {
        for (int k_w = 0; k_w < K; ++k_w) {
            int h_pad = h + k_h - pad;
            int w_pad = w + k_w - pad;
            if (h_pad >= 0 && h_pad < H && w_pad >= 0 && w_pad < W) {
                int index_col = n * H * W * C_in * K * K + h_pad * W * C_in * K * K + w_pad * C_in * K * K + c_in * K * K + (K-1-k_h) * K + (K-1-k_w);
                value += col[index_col];
            }
        }
    }
    im[index_im] = value;
}

void col2im_gpu(const float *col, float *im, const int Batchsize, const int C_in, const int H, const int W, const int K, const int pad) {
    int size = Batchsize * H * W * C_in;
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    col2im_kernel<<<grid_size, block_size>>>(col, im, Batchsize, C_in, H, W, K, pad);
}

void conv_layer_forward(const float *col, const float *Kernel, float *Y, const int N, const int C_in, const int H, const int W, const int C_out, const int K, const int pad) {
    
    float *right_shape_Y;
    cudaMalloc(&right_shape_Y, N * C_out * H * W * sizeof(float));
    // 将输入图像转换为矩阵

    // 执行卷积操作
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, Kernel, col, Y, C_out, C_in * K * K, N * H * W);

    // 将输出转换为原始形状
    change_shape_gpu(Y, right_shape_Y, N, C_out, H, W);

    cudaMemcpy(Y, right_shape_Y, N * C_out * H * W * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(right_shape_Y);
}

void conv_layer_backward_dKernel(const float *col, const float *dY, float *dKernel, const int N, const int C_in, const int H, const int W, const int C_out, const int K, const int pad) {
    float *d_dY;
    cudaMalloc(&d_dY, N * C_out * H * W * sizeof(float));
    reverse_change_shape_gpu(dY, d_dY, N, C_out, H, W);

    // 计算卷积核的梯度
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_T, d_dY, col, dKernel, C_out, N * H * W, C_in * K * K);

    // 释放 GPU 内存
    cudaFree(d_dY);
}


void conv_layer_backward_dX(const float *Kernel, const float *dY, float *dX, const int N, const int C_in, const int H, const int W, const int C_out, const int K, const int pad) {
    float *d_dY;
    float *d_dX;
    float *d_col;
    cudaMalloc(&d_dY, N * C_out * H * W * sizeof(float));
    cudaMalloc(&d_dX, N * C_in * H * W * sizeof(float));
    cudaMalloc(&d_col, N * H * W * C_in * K * K* sizeof(float));
    reverse_change_shape_gpu(dY, d_dY, N, C_out, H, W);

    // 计算输入的梯度
    gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_N, Kernel, d_dY, d_col, C_in * K * K, C_out, N * H * W);


    col2im_gpu(d_col, d_dX, N, C_in, H, W, K, pad);
    
    cudaMemcpy(dX, d_dX, N * C_in * H * W * sizeof(float), cudaMemcpyDeviceToDevice);

    // 释放 GPU 内存
    cudaFree(d_dY);
    cudaFree(d_dX);
    cudaFree(d_col);
}


__global__ void max_pooling_kernel(const float *input, float *output, const int N, const int C, const int H, const int W, const int Kernel_size, const int stride, float *mask) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C * (H / stride) * (W / stride)) {
        return;
    }
    int index_output = index;
    int w = index % (W / stride);
    index /= (W / stride);
    int h = index % (H / stride);
    index /= (H / stride);
    int c = index % C;
    index /= C;
    int n = index;

    float value = -1e38;
    int max_index = -1;
    for (int i = 0; i < Kernel_size; ++i) {
        for (int j = 0; j < Kernel_size; ++j) {
            int h_in = h * stride + i;
            int w_in = w * stride + j;
            int index_input = n * C * H * W + c * H * W + h_in * W + w_in;
            if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W && input[index_input] > value) {
                value = input[index_input];
                max_index = index_input;
            }
        }
    }
    output[index_output] = value;
    mask[index_output] = max_index;
    
}

void max_pooling_gpu(const float *input, float *output, const int N, const int C, const int H, const int W, const int Kernel_size, const int stride, float *mask) {
    int size = N * C * (H / stride) * (W / stride);
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    max_pooling_kernel<<<grid_size, block_size>>>(input, output, N, C, H, W, Kernel_size, stride, mask);
}

__global__ void max_pooling_backward_kernel(const float *dY, const float *mask, float *dX, const int N, const int C, const int H, const int W, const int Kernel_size, const int stride) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C * H * W) {
        return;
    }
    int index_input = index;
    int w = index % W;
    index /= W;
    int h = index % H;
    index /= H;
    int c = index % C;
    index /= C;
    int n = index;

    int h_out = h / stride;
    int w_out = w / stride;
    int index_output = n * C * (H / stride) * (W / stride) + c * (H / stride) * (W / stride) + h_out * (W / stride) + w_out;
    dX[index_input] = 0;
    if (mask[index_output] == index_input) {
        dX[index_input] = dY[index_output];
    }
}

void max_pooling_backward_gpu(const float *dY, const float *mask, float *dX, const int N, const int C, const int H, const int W, const int Kernel_size, const int stride) {
    int size = N * C * H * W;
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;

    max_pooling_backward_kernel<<<grid_size, block_size>>>(dY, mask, dX, N, C, H, W, Kernel_size, stride);
}


__global__ void softmax_max_kernel(const float *input, float *max_vals, const int N, const int C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) return;

    float max_val = -1e38;
    for (int c = 0; c < C; ++c) {
        max_val = fmaxf(max_val, input[index * C + c]);
    }
    max_vals[index] = max_val;
}

__global__ void softmax_subtract_max_kernel(const float *input, const float *max_vals, float *output, const int N, const int C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C) return;

    int n = index / C;
    output[index] = input[index] - max_vals[n];
}

__global__ void softmax_exp_kernel(const float *input, float *output, const int N, const int C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C) return;

    output[index] = expf(input[index]);
}

__global__ void softmax_sum_kernel(const float *input, float *sum_vals, const int N, const int C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) return;

    float sum_val = 0;
    for (int c = 0; c < C; ++c) {
        sum_val += input[index * C + c];
    }
    sum_vals[index] = sum_val;
}

__global__ void softmax_normalize_kernel(const float *input, const float *sum_vals, float *output, const int N, const int C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C) return;

    int n = index / C;
    output[index] = input[index] / sum_vals[n];
}

void softmax_gpu(const float *input, float *output, const int N, const int C) {
    float *d_max_vals, *d_sum_vals;
    cudaMalloc(&d_max_vals, N * sizeof(float));
    cudaMalloc(&d_sum_vals, N * sizeof(float));

    int block_size = 256;
    int grid_size_N = (N + block_size - 1) / block_size;
    int grid_size_NC = (N * C + block_size - 1) / block_size;

    softmax_max_kernel<<<grid_size_N, block_size>>>(input, d_max_vals, N, C);
    softmax_subtract_max_kernel<<<grid_size_NC, block_size>>>(input, d_max_vals, output, N, C);
    softmax_exp_kernel<<<grid_size_NC, block_size>>>(output, output, N, C);
    softmax_sum_kernel<<<grid_size_N, block_size>>>(output, d_sum_vals, N, C);
    softmax_normalize_kernel<<<grid_size_NC, block_size>>>(output, d_sum_vals, output, N, C);

    cudaFree(d_max_vals);
    cudaFree(d_sum_vals);
}

__global__ void cross_entropy_loss_kernel(const float *softmax_output, const int *labels, float *loss, const int N, const int C) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        int label = labels[index];
        atomicAdd(loss, -logf(softmax_output[index * C + label] + 1e-5) / N);
    }
}
__global__ void cross_entropy_loss_backward_kernel(const float *softmax_output, const int *labels, float *dX, const int N, const int C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C) return;

    int n = index / C;
    int c = index % C;
    int label = labels[n];

    if (c == label) {
        dX[index] = (softmax_output[index] - 1.0f) / N;
    } else {
        dX[index] = softmax_output[index] / N;
    }
}

void cross_entropy_loss_gpu(const float *softmax_output, const int *labels, float *loss, const int N, const int C) {
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    cross_entropy_loss_kernel<<<grid_size, block_size>>>(softmax_output, labels, loss, N, C);
}

void cross_entropy_loss_backward_gpu(const float *softmax_output, const int *labels, float *dX, const int N, const int C) {
    int block_size = 256;
    int grid_size = (N * C + block_size - 1) / block_size;

    cross_entropy_loss_backward_kernel<<<grid_size, block_size>>>(softmax_output, labels, dX, N, C);
}
Tensor::Tensor(std::vector<int> shape, bool device, std::vector<float> data, std::vector<float> grad) 
: shape(shape), onGPU(device){ 
size = 1;
for (int dim : shape) size *= dim;

cudaMalloc(&maxpooling_mask, size * sizeof(float));

cudaMalloc(&deviceGrad, size * sizeof(float)); 
cudaMalloc(&deviceData, size * sizeof(float));

cudaMemcpy(deviceData, data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(deviceGrad, grad.data(), size * sizeof(float), cudaMemcpyHostToDevice);
}

Tensor::Tensor(py::array_t<float> input_array) {
    onGPU = true;
    py::buffer_info buf_info = input_array.request();
    float *ptr = static_cast<float *>(buf_info.ptr);

    // 存储形状
    shape = std::vector<int>(buf_info.shape.begin(), buf_info.shape.end());
    size = 1;
    for (auto dim : shape) size *= dim;

    cudaMalloc(&maxpooling_mask, size * sizeof(float));

    cudaMalloc(&deviceGrad, size * sizeof(float)); 
    cudaMalloc(&deviceData, size * sizeof(float));

    cudaMemcpy(deviceData, ptr, size * sizeof(float), cudaMemcpyHostToDevice);
    
}

py::array_t<float> Tensor::data() {
    std::vector<float> result(size);
    if (onGPU) {
        cudaMemcpy(result.data(), deviceData, size * sizeof(float), cudaMemcpyDeviceToHost);
    } 

    // 将 std::vector<float> 转换为 numpy 数组
    py::array_t<float> array(result.size(), result.data());

    // 设置 numpy 数组的形状
    array.resize(shape);
    return array;
}

py::array_t<float> Tensor::grad() {
    std::vector<float> result(size);
    if (onGPU) {
        cudaMemcpy(result.data(), deviceGrad, size * sizeof(float), cudaMemcpyDeviceToHost);
    } 

    // 将 std::vector<float> 转换为 numpy 数组
    py::array_t<float> array(result.size(), result.data());

    // 设置 numpy 数组的形状
    array.resize(shape);
    return array;
}

std::vector<float> Tensor::get_data() {
    std::vector<float> result;
    if (onGPU) {
        float* hostData = new float[size]; // 在主机上分配内存
        cudaMemcpy(hostData, deviceData, size * sizeof(float), cudaMemcpyDeviceToHost);
        result.assign(hostData, hostData + size);
        
        delete[] hostData; 
    }
    return result;
}

std::vector<float> Tensor::get_grad() {
    std::vector<float> result;
    if (onGPU) {
        float* hostGrad = new float[size]; // 在主机上分配内存
        cudaMemcpy(hostGrad, deviceGrad, size * sizeof(float), cudaMemcpyDeviceToHost);
        result.assign(hostGrad, hostGrad + size);
        
        delete[] hostGrad; 
    }
    return result;
}

Tensor::~Tensor() {
    if (deviceData) {
        cudaFree(deviceData); 
    }
    if (deviceGrad) {
        cudaFree(deviceGrad); 
    }
    if (maxpooling_mask) {
        cudaFree(maxpooling_mask); 
    }
}

// float& Tensor::operator()(std::vector<int>& indices) {
//     size_t index = 0;
//     for (size_t i = 0; i < indices.size(); ++i) {
//         index = index * shape[i] + indices[i];
//     }
//     return data[index]; 
// }

// float& Tensor::grad_operator(std::vector<int> indices) {
//     size_t index = 0;
//     for (size_t i = 0; i < indices.size(); ++i) {
//         index = index * shape[i] + indices[i];
//     }
//     return grad[index]; 
// }

// Tensor& Tensor::cpu() {
//     if (onGPU) {
        
//         cudaMemcpy(data, deviceData, size * sizeof(float), cudaMemcpyDeviceToHost);
//         cudaMemcpy(grad, deviceGrad, size * sizeof(float), cudaMemcpyDeviceToHost);
//         onGPU = false;
//     }
//     return *this;
// }

// Tensor& Tensor::gpu() {
//     if (!onGPU) {
//         cudaMemcpy(deviceData, data, size * sizeof(float), cudaMemcpyHostToDevice);
//         onGPU = true;
//     }
//     return *this;
// }


// void Tensor::relu_cpu() {
    
//     for (size_t i = 0; i < size; ++i) {
//         data[i] = std::max(0.0f, data[i]);
//     }
// }

// void Tensor::sigmoid_cpu() {
    
//     for (size_t i = 0; i < size; ++i) {
//         data[i] = 1.0f / (1.0f + exp(-data[i]));
//     }
// }

void Tensor::relu_gpu() {
    
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock; 

    reluKernel<<<blocks, threadsPerBlock>>>(deviceData, size);
    cudaDeviceSynchronize(); 
}

void Tensor::sigmoid_gpu() {
    
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock; 

    sigmoidKernel<<<blocks, threadsPerBlock>>>(deviceData, size);
    cudaDeviceSynchronize(); 
}

void Tensor::relu_backward_gpu(float* gradOutput) {

    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    relu_backwardKernel<<<blocks, threadsPerBlock>>>(deviceData, gradOutput, deviceGrad, size);
    cudaDeviceSynchronize(); 
}


void Tensor::sigmoid_backward_gpu(float* gradOutput) {
    int threadsPerBlock = 256;
    int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;

    sigmoid_backwardKernel<<<blocks, threadsPerBlock>>>(deviceData, gradOutput, deviceGrad, size);
    cudaDeviceSynchronize();  
};


Fully_Connected::Fully_Connected(int input_size, int output_size, bool onGPU, std::vector<float> data):
input_size(input_size), output_size(output_size), onGPU(onGPU),
weight({output_size, input_size}, onGPU, data) {
    cublasStatus_t status = cublasCreate(&handle);
}

Fully_Connected::Fully_Connected(py::array_t<float> input_array):
weight(input_array) {
    onGPU = true;
    py::buffer_info buf_info = input_array.request();
    float *ptr = static_cast<float *>(buf_info.ptr);

    // 存储形状
    output_size = buf_info.shape[0];
    input_size = buf_info.shape[1];

    cublasStatus_t status = cublasCreate(&handle);
    
    
}

void Fully_Connected::forward(Tensor &X, Tensor &Y) {
    int N = X.shape[0];
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, weight.deviceData, X.deviceData, Y.deviceData, output_size, input_size, N);
 
}

void Fully_Connected::backward(Tensor &X, Tensor &Y) {
    int N = X.shape[0];
    gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_N, weight.deviceData, Y.deviceGrad, X.deviceGrad, input_size, output_size, N);
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_T, Y.deviceGrad, X.deviceData, weight.deviceGrad, output_size, N, input_size);
}


Convolutional::Convolutional(int in_channels, int out_channels, int kernel_size,int padding, bool onGPU, std::vector<float> data):
in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), padding(padding), onGPU(onGPU),
weight({out_channels, in_channels, kernel_size, kernel_size}, onGPU, data) {
    cublasStatus_t status = cublasCreate(&handle);
}

void Convolutional::forward(Tensor &X, Tensor &Y) {
    int N = X.shape[0];
    int H = X.shape[2];
    int W = X.shape[3];
    int C_in = X.shape[1];
    int H_out = H - kernel_size + 2 * padding + 1;
    int W_out = W - kernel_size + 2 * padding + 1;
    int K = kernel_size;
    int C_out = out_channels;

    float *d_col;
    cudaMalloc(&d_col, N * H * W * C_in * K * K * sizeof(float));
    im2col_gpu(X.deviceData, d_col, N, C_in, H, W, K, padding);

    conv_layer_forward(d_col, weight.deviceData, Y.deviceData, N, C_in, H, W, C_out, K, padding); 
    cudaFree(d_col);
}

void Convolutional::backward(Tensor &X, Tensor &Y) {
    int N = X.shape[0];
    int H = X.shape[2];
    int W = X.shape[3];
    int C_in = X.shape[1];
    int H_out = H - kernel_size + 2 * padding + 1;
    int W_out = W - kernel_size + 2 * padding + 1;
    int K = kernel_size;
    int C_out = out_channels;

    float *d_col;
    cudaMalloc(&d_col, N * H * W * C_in * K * K * sizeof(float));
    im2col_gpu(X.deviceData, d_col, N, C_in, H, W, K, padding);

    conv_layer_backward_dKernel(d_col, Y.deviceGrad, weight.deviceGrad, N, C_in, H, W, C_out, K, padding);
    conv_layer_backward_dX(weight.deviceData, Y.deviceGrad, X.deviceGrad, N, C_in, H, W, C_out, K, padding);
    cudaFree(d_col);
}

MaxPooling::MaxPooling(int kernel_size, bool onGPU):
kernel_size(kernel_size), onGPU(onGPU) {
}


void MaxPooling::forward(Tensor &X, Tensor &Y) {
    int N = X.shape[0];
    int C = X.shape[1];
    int H = X.shape[2];
    int W = X.shape[3];
    int stride = kernel_size;


    max_pooling_gpu(X.deviceData, Y.deviceData, N, C, H, W, kernel_size, stride, Y.maxpooling_mask);
}

void MaxPooling::backward(Tensor &X, Tensor &Y) {
    int N = X.shape[0];
    int C = X.shape[1];
    int H = X.shape[2];
    int W = X.shape[3];
    int stride = kernel_size;

    max_pooling_backward_gpu(Y.deviceGrad, Y.maxpooling_mask, X.deviceGrad, N, C, H, W, kernel_size, stride);
}

py::array_t<float> Softmax(Tensor& input){
    int N = input.shape[0];
    int C = input.shape[1];
    float* output;
    cudaMalloc(&output, N * C * sizeof(float));
    softmax_gpu(input.deviceData, output, N, C);
    std::vector<int> shape = {N, C};
    py::array_t<float> result(shape);
    cudaMemcpy(result.mutable_data(), output, N * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(output);
    return result;
}

py::array_t<float> CrossEntropyLoss(Tensor& softmax_output, Tensor& labels){
    int N = softmax_output.shape[0];
    int C = softmax_output.shape[1];
    float* loss;
    cudaMalloc(&loss, sizeof(float));
    cudaMemset(loss, 0, sizeof(float));
    // 分配设备内存并转换 labels
    int* labels_device;
    cudaMalloc(&labels_device, N * sizeof(int));

    // 将 float 类型的 labels 转换为 int
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    float_to_int_kernel<<<blocksPerGrid, threadsPerBlock>>>(labels.deviceData, labels_device, N);

    cross_entropy_loss_gpu(softmax_output.deviceData, labels_device, loss, N, C);

    cudaFree(labels_device);
    std::vector<int> shape = {1};
    py::array_t<float> result(shape);
    cudaMemcpy(result.mutable_data(), loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(loss);
    return result;
}

py::array_t<float> CrossEntropyLoss_backward_with_Softmax(Tensor& softmax_input, Tensor& labels){
    int N = softmax_input.shape[0];
    int C = softmax_input.shape[1];
    float* dX;
    cudaMalloc(&dX, N * C * sizeof(float));
    // 分配设备内存并转换 labels
    int* labels_device;
    cudaMalloc(&labels_device, N * sizeof(int));

    // 将 float 类型的 labels 转换为 int
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    float_to_int_kernel<<<blocksPerGrid, threadsPerBlock>>>(labels.deviceData, labels_device, N);
    
    //做softmax
    float* softmax_output;
    cudaMalloc(&softmax_output, N * C * sizeof(float));
    softmax_gpu(softmax_input.deviceData, softmax_output, N, C);
    cross_entropy_loss_backward_gpu(softmax_output, labels_device, dX, N, C);

    cudaFree(softmax_output);
    cudaFree(labels_device);
    std::vector<int> shape = {N, C};
    py::array_t<float> result(shape);
    cudaMemcpy(result.mutable_data(), dX, N * C * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dX);
    return result;
}


py::array_t<float> fc_forward(Tensor& input, Tensor& weight){
    int N = input.shape[0];
    int input_size = input.shape[1];
    int output_size = weight.shape[1];
    float *output;
    cudaMalloc(&output, N * output_size * sizeof(float));
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, weight.deviceData, input.deviceData, output, output_size, input_size, N);
    std::vector<int> shape = {N, output_size};
    py::array_t<float> result(shape);
    cudaMemcpy(result.mutable_data(), output, N * output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(output);
    return result;
}

py::array_t<float> conv_forward(Tensor& input, Tensor& weight){
    int kernel_size = 3;
    int padding = 1;
    int out_channels = weight.shape[0];
    int N = input.shape[0];
    int H = input.shape[2];
    int W = input.shape[3];
    int C_in = input.shape[1];
    int H_out = H - kernel_size + 2 * padding + 1;
    int W_out = W - kernel_size + 2 * padding + 1;
    int K = kernel_size;
    int C_out = out_channels;

    float *d_col;
    cudaMalloc(&d_col, N * H * W * C_in * K * K * sizeof(float));
    im2col_gpu(input.deviceData, d_col, N, C_in, H, W, K, padding);
    float *output;
    cudaMalloc(&output, N * C_out * H_out * W_out * sizeof(float));
    conv_layer_forward(d_col, weight.deviceData, output, N, C_in, H, W, C_out, K, padding); 
    cudaFree(d_col);
    std::vector<int> shape = {N, C_out, H_out, W_out};
    py::array_t<float> result(shape);
    cudaMemcpy(result.mutable_data(), output, N * C_out * H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(output);
    return result;
}

std::tuple<py::array_t<float>, py::array_t<float>> maxpooling_forward(Tensor& input){
    int kernel_size = 2;
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    int stride = kernel_size;

    float *output;
    cudaMalloc(&output, N * C * (H / stride) * (W / stride) * sizeof(float));
    float *mask;
    cudaMalloc(&mask, N * C * (H / stride) * (W / stride) * sizeof(float));

    max_pooling_gpu(input.deviceData, output, N, C, H, W, kernel_size, stride, mask);
    std::vector<int> shape1 = {N, C, H / stride, W / stride};
    py::array_t<float> result1(shape1);
    cudaMemcpy(result1.mutable_data(), output, N * C * (H / stride) * (W / stride) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(output);
    std::vector<int> shape2 = {N, C, H / stride, W / stride};
    py::array_t<float> result2(shape2);
    cudaMemcpy(result2.mutable_data(), mask, N * C * (H / stride) * (W / stride) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(mask);
    return std::make_tuple(result1, result2);
}

std::tuple<py::array_t<float>, py::array_t<float>> fc_backward(Tensor& input, Tensor& weight, Tensor& gradOutput){
    int N = input.shape[0];
    int input_size = input.shape[1];
    int output_size = weight.shape[1];
    float *dX, *dWeight;
    cudaMalloc(&dX, N * input_size * sizeof(float));
    cudaMalloc(&dWeight, output_size * input_size * sizeof(float));
    gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_N, weight.deviceData, gradOutput.deviceData, dX, input_size, output_size, N);
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_T, gradOutput.deviceData, input.deviceData, dWeight, output_size, N, input_size);
    std::vector<int> shape1 = {N, input_size};
    py::array_t<float> result1(shape1);
    cudaMemcpy(result1.mutable_data(), dX, N * input_size * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<int> shape2 = {input_size, output_size};
    py::array_t<float> result2(shape2);
    cudaMemcpy(result2.mutable_data(), dWeight, output_size * input_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dX);
    cudaFree(dWeight);
    return std::make_tuple(result1, result2);
}


// std::tuple<py::array_t<float>, py::array_t<float>> conv_backward(Tensor& input, Tensor& weight, Tensor& gradOutput) {
//     std::cout << "Starting conv_backward..." << std::endl;

//     int kernel_size = 3;
//     int padding = 1;
//     int out_channels = weight.shape[0];
//     int N = input.shape[0];
//     int H = input.shape[2];
//     int W = input.shape[3];
//     int C_in = input.shape[1];
//     int H_out = H - kernel_size + 2 * padding + 1;
//     int W_out = W - kernel_size + 2 * padding + 1;
//     int K = kernel_size;
//     int C_out = out_channels;

//     std::cout << "Parameters: N=" << N << ", C_in=" << C_in << ", H=" << H << ", W=" << W
//               << ", C_out=" << C_out << ", K=" << K << ", padding=" << padding << std::endl;

//     // Allocate memory for d_col
//     float *d_col = nullptr;
//     cudaError_t err = cudaMalloc(&d_col, N * H * W * C_in * K * K * sizeof(float));
//     if (err != cudaSuccess) {
//         std::cerr << "Failed to allocate memory for d_col: " << cudaGetErrorString(err) << std::endl;
//         throw std::runtime_error("CUDA memory allocation failed");
//     }
//     std::cout << "Allocated memory for d_col." << std::endl;

//     // Call im2col_gpu
//     std::cout << "Calling im2col_gpu..." << std::endl;
//     im2col_gpu(input.deviceData, d_col, N, C_in, H, W, K, padding);
//     cudaDeviceSynchronize();
//     std::cout << "im2col_gpu completed." << std::endl;

//     // Allocate memory for dX and dWeight
//     float *dX = nullptr;
//     float *dWeight = nullptr;
//     err = cudaMalloc(&dX, N * C_in * H * W * sizeof(float));
//     if (err != cudaSuccess) {
//         std::cerr << "Failed to allocate memory for dX: " << cudaGetErrorString(err) << std::endl;
//         cudaFree(d_col);
//         throw std::runtime_error("CUDA memory allocation failed");
//     }
//     err = cudaMalloc(&dWeight, C_out * C_in * K * K * sizeof(float));
//     if (err != cudaSuccess) {
//         std::cerr << "Failed to allocate memory for dWeight: " << cudaGetErrorString(err) << std::endl;
//         cudaFree(d_col);
//         cudaFree(dX);
//         throw std::runtime_error("CUDA memory allocation failed");
//     }
//     std::cout << "Allocated memory for dX and dWeight." << std::endl;

//     // Call conv_layer_backward_dKernel
//     std::cout << "Calling conv_layer_backward_dKernel..." << std::endl;
//     conv_layer_backward_dKernel(d_col, gradOutput.deviceData, dWeight, N, C_in, H, W, C_out, K, padding);
//     cudaDeviceSynchronize();
//     std::cout << "conv_layer_backward_dKernel completed." << std::endl;

//     // Call conv_layer_backward_dX
//     std::cout << "Calling conv_layer_backward_dX..." << std::endl;
//     conv_layer_backward_dX(weight.deviceData, gradOutput.deviceData, dX, N, C_in, H, W, C_out, K, padding);
//     cudaDeviceSynchronize();
//     std::cout << "conv_layer_backward_dX completed." << std::endl;

//     // Free d_col
//     cudaFree(d_col);
//     std::cout << "Freed memory for d_col." << std::endl;

//     // Copy dX to host
//     std::vector<int> shape1 = {N, C_in, H, W};
//     py::array_t<float> result1(shape1);
//     // err = cudaMemcpy(result1.mutable_data(), dX, N * C_in * H * W * sizeof(float), cudaMemcpyDeviceToHost);
//     // if (err != cudaSuccess) {
//     //     std::cerr << "Failed to copy dX to host: " << cudaGetErrorString(err) << std::endl;
//     //     cudaFree(dX);
//     //     cudaFree(dWeight);
//     //     throw std::runtime_error("CUDA memory copy failed");
//     // }
//     // std::cout << "Copied dX to host." << std::endl;

//     // Copy dWeight to host
//     std::vector<int> shape2 = {C_out, C_in, K, K};
//     py::array_t<float> result2(shape2);
//     err = cudaMemcpy(result2.mutable_data(), dWeight, C_out * C_in * K * K * sizeof(float), cudaMemcpyDeviceToHost);
//     if (err != cudaSuccess) {
//         std::cerr << "Failed to copy dWeight to host: " << cudaGetErrorString(err) << std::endl;
//         cudaFree(dX);
//         cudaFree(dWeight);
//         throw std::runtime_error("CUDA memory copy failed");
//     }
//     std::cout << "Copied dWeight to host." << std::endl;

//     // Free dX and dWeight
//     cudaFree(dX);
//     cudaFree(dWeight);
//     std::cout << "Freed memory for dX and dWeight." << std::endl;

//     std::cout << "conv_backward completed successfully." << std::endl;
//     return std::make_tuple(result1, result2);
// }


std::tuple<py::array_t<float>, py::array_t<float>> conv_backward(Tensor& input, Tensor& weight, Tensor& gradOutput) {

    int kernel_size = 3;
    int padding = 1;
    int out_channels = weight.shape[0];
    int N = input.shape[0];
    int H = input.shape[2];
    int W = input.shape[3];
    int C_in = input.shape[1];
    int H_out = H - kernel_size + 2 * padding + 1;
    int W_out = W - kernel_size + 2 * padding + 1;
    int K = kernel_size;
    int C_out = out_channels;

    // Allocate memory for d_col
    float *d_col = nullptr;
    cudaError_t err = cudaMalloc(&d_col, N * H * W * C_in * K * K * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate memory for d_col: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA memory allocation failed");
    }

    im2col_gpu(input.deviceData, d_col, N, C_in, H, W, K, padding);
    cudaDeviceSynchronize();

    // Allocate memory for dX and dWeight
    float *dX = nullptr, *dWeight = nullptr;
    err = cudaMalloc(&dX, N * C_in * H * W * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate memory for dX: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_col);
        throw std::runtime_error("CUDA memory allocation failed");
    }
    err = cudaMalloc(&dWeight, C_out * C_in * K * K * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate memory for dWeight: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_col);
        cudaFree(dX);
        throw std::runtime_error("CUDA memory allocation failed");
    }

    // Call conv_layer_backward_dKernel
    conv_layer_backward_dKernel(d_col, gradOutput.deviceData, dWeight, N, C_in, H, W, C_out, K, padding);
    cudaDeviceSynchronize();

    // Call conv_layer_backward_dX
    conv_layer_backward_dX(weight.deviceData, gradOutput.deviceData, dX, N, C_in, H, W, C_out, K, padding);
    cudaDeviceSynchronize();

    // Free d_col
    cudaFree(d_col);

    // Copy dX to host
    std::vector<int> shape1 = {N, C_in, H, W};
    py::array_t<float> result1(shape1);
    err = cudaMemcpy(result1.mutable_data(), dX, N * C_in * H * W * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy dX to host: " << cudaGetErrorString(err) << std::endl;
        cudaFree(dX);
        cudaFree(dWeight);
        throw std::runtime_error("CUDA memory copy failed");
    }

    // Copy dWeight to host
    std::vector<int> shape2 = {C_out, C_in, K, K};
    py::array_t<float> result2(shape2);
    err = cudaMemcpy(result2.mutable_data(), dWeight, C_out * C_in * K * K * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy dWeight to host: " << cudaGetErrorString(err) << std::endl;
        cudaFree(dX);
        cudaFree(dWeight);
        throw std::runtime_error("CUDA memory copy failed");
    }

    // Free dX and dWeight
    cudaFree(dX);
    cudaFree(dWeight);
    return std::make_tuple(result1, result2);
}

py::array_t<float> maxpooling_backward(Tensor& input, Tensor& gradOutput, Tensor& mask) {
    int kernel_size = 2;
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    int stride = kernel_size;

    float *output;
    cudaMalloc(&output, N * C * H * W * sizeof(float));
    max_pooling_backward_gpu(gradOutput.deviceData, mask.deviceData, output, N, C, H, W, kernel_size, stride);
    std::vector<int> shape = {N, C, H, W};
    py::array_t<float> result(shape);
    cudaMemcpy(result.mutable_data(), output, N * C * H * W * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(output);
    return result;
}

