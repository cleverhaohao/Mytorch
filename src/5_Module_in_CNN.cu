#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include "Tensor.h"
#include "5_Module_in_CNN.h" 
using namespace std;
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


void fully_connected_layer_forward(const float *X, const float *W, float *Y, const int N, const int C_in, const int C_out) {
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_N, W, X, Y, C_out, C_in, N);
}


void fully_connected_layer_backward_dW(const float *X, const float *dY, float *dW, const int N, const int C_in, const int C_out) {
    // 计算权重矩阵的梯度 dW = X^T * dY
    gemm_gpu(CUBLAS_OP_N, CUBLAS_OP_T, dY, X, dW, C_out, N, C_in);
}

void fully_connected_layer_backward_dX(const float *dY, const float *W, float *dX, const int N, const int C_in, const int C_out) {
    // 计算输入的梯度 dX = dY * W^T
    gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_N, W, dY, dX, C_in, C_out, N);
}


void fully_connected_layer_cpu(const float *h_X, const float *h_W, float *h_Y, const float *h_dY, float *h_dW, float *h_dX, const int N, const int C_in, const int C_out) {
    // 分配 GPU 内存
    float *d_X, *d_W, *d_Y, *d_dY, *d_dW, *d_dX;
    cudaMalloc(&d_X, N * C_in * sizeof(float));
    cudaMalloc(&d_W, C_in * C_out * sizeof(float));
    cudaMalloc(&d_Y, N * C_out * sizeof(float));
    cudaMalloc(&d_dY, N * C_out * sizeof(float));
    cudaMalloc(&d_dW, C_in * C_out * sizeof(float));
    cudaMalloc(&d_dX, N * C_in * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_X, h_X, N * C_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W, C_in * C_out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dY, h_dY, N * C_out * sizeof(float), cudaMemcpyHostToDevice);

    // 正向传播
    fully_connected_layer_forward(d_X, d_W, d_Y, N, C_in, C_out);

    // 反向传播
    fully_connected_layer_backward_dW(d_X, d_dY, d_dW, N, C_in, C_out);

    fully_connected_layer_backward_dX(d_dY, d_W, d_dX, N, C_in, C_out);

    // 将结果从设备复制回主机
    cudaMemcpy(h_Y, d_Y, N * C_out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dW, d_dW, C_in * C_out * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dX, d_dX, N * C_in * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_Y);
    cudaFree(d_dY);
    cudaFree(d_dW);
    cudaFree(d_dX);

    std::cout << "Forward Output Y:" << std::endl;
    for (int i = 0; i < N * C_out; ++i) {
        std::cout << h_Y[i] << " ";
        if ((i + 1) % C_out == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    

    std::cout << "Backward dW:" << std::endl;
    for (int i = 0; i < C_in * C_out; ++i) {
        std::cout << h_dW[i] << " ";
        if ((i + 1) % C_out == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    std::cout << "Backward dX:" << std::endl;
    for (int i = 0; i < N * C_in; ++i) {
        std::cout << h_dX[i] << " ";
        if ((i + 1) % C_in == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

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
    float *d_dY, *d_dX;
    cudaMalloc(&d_dY, N * C_out * H * W * sizeof(float));
    cudaMalloc(&d_dX, N * C_in * H * W * sizeof(float));
    reverse_change_shape_gpu(dY, d_dY, N, C_out, H, W);

    // 计算输入的梯度
    gemm_gpu(CUBLAS_OP_T, CUBLAS_OP_N, Kernel, d_dY, dX, C_in * K * K, C_out, N * H * W);


    col2im_gpu(dX, d_dX, N, C_in, H, W, K, pad);
    
    cudaMemcpy(dX, d_dX, N * C_in * H * W * sizeof(float), cudaMemcpyDeviceToDevice);

    // 释放 GPU 内存
    cudaFree(d_dY);
    cudaFree(d_dX);
}

void conv_layer_cpu(const float *h_X, const float *h_Kernel, float *h_Y, const float *h_dY, float *h_dKernel, float *h_dX, const int N, const int C_in, const int H, const int W, const int C_out, const int K, const int pad) {
    // 分配 GPU 内存
    float *d_X, *d_Kernel, *d_Y, *d_dY, *d_dKernel, *d_dX;
    cudaMalloc(&d_X, N * C_in * H * W * sizeof(float));
    cudaMalloc(&d_Kernel, C_out * C_in * K * K * sizeof(float));
    cudaMalloc(&d_Y, N * C_out * H * W * sizeof(float));
    cudaMalloc(&d_dY, N * C_out * H * W * sizeof(float));
    cudaMalloc(&d_dKernel, C_out * C_in * K * K * sizeof(float));
    cudaMalloc(&d_dX, N * C_in * H * W * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_X, h_X, N * C_in * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Kernel, h_Kernel, C_out * C_in * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dY, h_dY, N * C_out * H * W * sizeof(float), cudaMemcpyHostToDevice);

    // 将输入图像转换为矩阵
    float *d_col;
    cudaMalloc(&d_col, N * H * W * C_in * K * K * sizeof(float));
    im2col_gpu(d_X, d_col, N, C_in, H, W, K, pad);

    // 正向传播
    conv_layer_forward(d_col, d_Kernel, d_Y, N, C_in, H, W, C_out, K, pad);

    // 反向传播
    conv_layer_backward_dKernel(d_col, d_dY, d_dKernel, N, C_in, H, W, C_out, K, pad);
    conv_layer_backward_dX(d_Kernel, d_dY, d_dX, N, C_in, H, W, C_out, K, pad);

    // 将结果从设备复制回主机
    cudaMemcpy(h_Y, d_Y, N * C_out * H * W * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dKernel, d_dKernel, C_out * C_in * K * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dX, d_dX, N * C_in * H * W * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_X);
    cudaFree(d_Kernel);
    cudaFree(d_Y);
    cudaFree(d_dY);
    cudaFree(d_dKernel);
    cudaFree(d_dX);

    std::cout << "Forward Output Y:" << std::endl;
    for (int i = 0; i < N * C_out * H * W; ++i) {
        std::cout << h_Y[i] << " ";
        if ((i + 1) % (C_out * H * W) == 0) {
            std::cout << std::endl;
        }
    }

    std::cout << "Backward dKernel:" << std::endl;
    for (int i = 0; i < C_out * C_in * K * K; ++i) {
        std::cout << h_dKernel[i] << " ";
        if ((i + 1) % (C_out * K * K) == 0) {
            std::cout << std::endl;
        }
    }

    std::cout << "Backward dX:" << std::endl;
    for (int i = 0; i < N * C_in * H * W; ++i) {
        std::cout << h_dX[i] << " ";
        if ((i + 1) % (C_in * H * W) == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;


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

void max_pooling_cpu(const float *h_X, float *h_Y, const float *h_dY, float *h_dX, const int N, const int C, const int H, const int W, const int Kernel_size, const int stride) {
    // 分配 GPU 内存
    float *d_X, *d_Y, *d_dY, *d_dX, *mask;
    cudaMalloc(&d_X, N * C * H * W * sizeof(float));
    cudaMalloc(&d_Y, N * C * (H / stride) * (W / stride) * sizeof(float));
    cudaMalloc(&d_dY, N * C * (H / stride) * (W / stride) * sizeof(float));
    cudaMalloc(&d_dX, N * C * H * W * sizeof(float));
    cudaMalloc(&mask, N * C * (H / stride) * (W / stride) * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_X, h_X, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dY, h_dY, N * C * (H / stride) * (W / stride) * sizeof(float), cudaMemcpyHostToDevice);

    // 正向传播
    max_pooling_gpu(d_X, d_Y, N, C, H, W, Kernel_size, stride, mask);

    // 反向传播
    max_pooling_backward_gpu(d_dY, mask, d_dX, N, C, H, W, Kernel_size, stride);

    // 将结果从设备复制回主机
    cudaMemcpy(h_Y, d_Y, N * C * (H / stride) * (W / stride) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dX, d_dX, N * C * H * W * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_dY);
    cudaFree(d_dX);
    cudaFree(mask);

    std::cout << "Forward Output Y:" << std::endl;
    for (int i = 0; i < N * C * (H / stride) * (W / stride); ++i) {
        std::cout << h_Y[i] << " ";
        if ((i + 1) % (C * (H / stride) * (W / stride)) == 0) {
            std::cout << std::endl;
        }
    }

    std::cout << "Backward dX:" << std::endl;
    for (int i = 0; i < N * C * H * W; ++i) {
        std::cout << h_dX[i] << " ";
        if ((i + 1) % (C * H * W) == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
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

void softmax_cpu(const float *h_X, float *h_Y, const int N, const int C) {
    // 分配 GPU 内存
    float *d_X, *d_Y;
    cudaMalloc(&d_X, N * C * sizeof(float));
    cudaMalloc(&d_Y, N * C * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_X, h_X, N * C * sizeof(float), cudaMemcpyHostToDevice);

    // 执行 softmax 操作
    softmax_gpu(d_X, d_Y, N, C);

    // 将结果从设备复制回主机
    cudaMemcpy(h_Y, d_Y, N * C * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_X);
    cudaFree(d_Y);

    std::cout << "Output Y:" << std::endl;
    for (int i = 0; i < N * C; ++i) {
        std::cout << h_Y[i] << " ";
        if ((i + 1) % C == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}


__global__ void cross_entropy_loss_kernel(const float *softmax_output, const int *labels, float *loss, const int N, const int C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N) return;

    int label = labels[index];
    loss[index] = -logf(softmax_output[index * C + label]);
}

__global__ void cross_entropy_loss_backward_kernel(const float *softmax_output, const int *labels, float *dX, const int N, const int C) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C) return;

    int n = index / C;
    int c = index % C;
    int label = labels[n];

    if (c == label) {
        dX[index] = softmax_output[index] - 1.0f;
    } else {
        dX[index] = softmax_output[index];
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

void cross_entropy_loss_with_softmax_cpu(const float *h_softmax_input, const int *h_labels, float *h_loss, float *h_dX, const int N, const int C) {
// 分配 GPU 内存
float *d_softmax_input, *d_softmax_output, *d_loss, *d_dX;
int *d_labels;
cudaMalloc(&d_softmax_input, N * C * sizeof(float));
cudaMalloc(&d_softmax_output, N * C * sizeof(float));
cudaMalloc(&d_loss, N * sizeof(float));
cudaMalloc(&d_dX, N * C * sizeof(float));
cudaMalloc(&d_labels, N * sizeof(int));

// 将数据从主机复制到设备
cudaMemcpy(d_softmax_input, h_softmax_input, N * C * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_labels, h_labels, N * sizeof(int), cudaMemcpyHostToDevice);

// 计算 softmax
softmax_gpu(d_softmax_input, d_softmax_output, N, C);

// 计算交叉熵损失
cross_entropy_loss_gpu(d_softmax_output, d_labels, d_loss, N, C);

// 计算交叉熵损失的梯度
cross_entropy_loss_backward_gpu(d_softmax_output, d_labels, d_dX, N, C);

// 将结果从设备复制回主机
cudaMemcpy(h_loss, d_loss, N * sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(h_dX, d_dX, N * C * sizeof(float), cudaMemcpyDeviceToHost);

// 释放 GPU 内存
cudaFree(d_softmax_input);
cudaFree(d_softmax_output);
cudaFree(d_loss);
cudaFree(d_dX);
cudaFree(d_labels);

std::cout << "Cross Entropy Loss:" << std::endl;
for (int i = 0; i < N; ++i) {
    std::cout << h_loss[i] << " ";
}
std::cout << std::endl;

std::cout << "Gradient dX:" << std::endl;
for (int i = 0; i < N * C; ++i) {
    std::cout << h_dX[i] << " ";
    if ((i + 1) % C == 0) {
        std::cout << std::endl;
    }
}
std::cout << std::endl;
}


// int main() {
    
    // const int N = 2;  // batch size
    // const int C_in = 3;  // input channels
    // const int C_out = 4;  // output channels

    // // 初始化输入 X 和权重 W
    // float X[N * C_in] = {1, 2, 3, 4, 5, 6};
    // float W[C_in * C_out] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    // // 正向传播结果
    // float Y[N * C_out];

    // // 反向传播结果
    // float dY[N * C_out] = {1, 2, 3, 4, 5, 6, 7, 8};  // 假设的 dY
    // float dW[C_in * C_out];
    // float dX[N * C_in];

    // // 调用新的函数进行计算
    // fully_connected_layer_cpu(X, W, Y, dY, dW, dX, N, C_in, C_out);
   




    // const int N = 2;  // batch size
    // const int C_in = 2;  // input channels
    // const int H = 3;  // height of input
    // const int W = 3;  // width of input
    // const int C_out = 1;  // output channels
    // const int K = 3;  // kernel size
    // const int pad = 1;  // padding
    // // 初始化输入 X 和卷积核 Kernel
    // float X[N * C_in * H * W] = {
    //     1, 2, 3, 4, 5, 6, 7, 8, 9,
    //     10, 11, 12, 13, 14, 15, 16, 17, 18,
    //     19, 20, 21, 22, 23, 24, 25, 26, 27,
    //     28, 29, 30, 31, 32, 33, 34, 35, 36
    // };
    // float Kernel[C_out * C_in * K * K] = {
    //     1, 0, -1,
    //     1, 0, -1,
    //     1, 0, -1,
    //     1, 0, -1,
    //     1, 0, -1,
    //     1, 0, -1
    // };
    // // 正向传播结果
    // float Y[N * C_out * H * W];
    // // 反向传播结果
    // float dY[N * C_out * H * W] = {
    //     1, 1, 1,
    //     1, 1, 1,
    //     1, 1, 1,
    //     1, 1, 1,
    //     1, 1, 1,
    //     1, 1, 1
    // };  // 假设的 dY
    // float dKernel[C_out * C_in * K * K];
    // float dX[N * C_in * H * W];
    // // 调用新的函数进行计算
    // conv_layer_cpu(X, Kernel, Y, dY, dKernel, dX, N, C_in, H, W, C_out, K, pad);




    // const int N = 2;  // batch size
    // const int C = 2;  // channels
    // const int H = 4;  // height of input
    // const int W = 4;  // width of input
    // const int Kernel_size = 2;  // kernel size
    // const int stride = 2;  // stride
    // // 初始化输入 X
    // float X[N * C * H * W] = {
    //     1, 2, 3, 4,
    //     5, 6, 7, 8,
    //     9, 10, 11, 12,
    //     13, 14, 15, 16,
    //     17, 18, 19, 20,
    //     21, 22, 23, 24,
    //     25, 26, 27, 28,
    //     29, 30, 31, 32
    // };
    // // 正向传播结果
    // float Y[N * C * (H / stride) * (W / stride)];
    // // 反向传播结果
    // float dY[N * C * (H / stride) * (W / stride)] = {
    //     1, 1,
    //     1, 1,
    //     1, 1,
    //     1, 1,
    //     1, 1,
    //     1, 1,
    //     1, 1,
    //     1, 1
    // };  // 假设的 dY
    // float dX[N * C * H * W];
    // // 调用新的函数进行计算
    // max_pooling_cpu(X, Y, dY, dX, N, C, H, W, Kernel_size, stride);





    // const int N = 3;  // batch size
    // const int C = 4;  // number of classes

    // // Initialize input X
    // float X[N * C] = {
    //     1.5, 2.0, 3.0, 4.0,
    //     2.0, 3.0, 3.0, 5.0,
    //     3.0, 4.0, 5.0, 8.0
    // };

    // // Output Y
    // float Y[N * C];

    // // Call the softmax function
    // softmax_cpu(X, Y, N, C);



    
//     const int N = 3;  // batch size
//     const int C = 4;  // number of classes

//     // Initialize input softmax_input
//     float softmax_input[N * C] = {
//         1.5, 2.0, 3.0, 4.0,
//         2.0, 3.0, 3.0, 5.0,
//         3.0, 4.0, 5.0, 8.0
//     };

//     // Initialize labels
//     int labels[N] = {3, 2, 1};

//     // Output loss
//     float loss[N];

//     // Gradient dX
//     float dX[N * C];

//     // Call the cross entropy loss with softmax function
//     cross_entropy_loss_with_softmax_cpu(softmax_input, labels, loss, dX, N, C);
   
// }


Fully_Connected::Fully_Connected(int input_size, int output_size, bool onGPU):
input_size(input_size), output_size(output_size), onGPU(onGPU) {
    weight = Tensor({input_size, output_size}, onGPU);
    grad_weight = Tensor({input_size, output_size}, onGPU);
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


// Convolution::Convolution(int input_channels_, int output_channels_, int kernel_size_, int stride_, int padding_, bool onGPU_) {
//     input_channels = input_channels_;
//     output_channels = output_channels_;
//     kernel_size = kernel_size_;
//     stride = stride_;
//     padding = padding_;
//     onGPU = onGPU_;
//     weight = new Tensor({output_channels, input_channels, kernel_size, kernel_size}, onGPU);
//     grad_weight = new Tensor({output_channels, input_channels, kernel_size, kernel_size}, onGPU);
// }