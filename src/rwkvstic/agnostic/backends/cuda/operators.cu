#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#include <cuda_fp16.h>
#define MIN_VALUE (-1e38)
typedef at::Half fp16;

__half *cast(fp16 *ptr)
{
    return reinterpret_cast<__half *>(ptr);
}

__global__ void kernel_mm8_seq(
    const int B, const int N, const int M,
    const __half *__restrict__ const x, const int x_stride,
    const uint8_t *__restrict__ const w, const int w_stride,
    __half *__restrict__ const y, const int y_stride)
{

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < B && k < M)
    {
        float y_local = 0;
        for (int j = 0; j < N; ++j)
        {
            y_local += __half2float(x[i * x_stride + j]) * (float(w[k * w_stride + j]));
        }
        y[i * y_stride + k] = __float2half(y_local);
    }
}
void cuda_mm8_seq(int B, int N, int M,
                  fp16 *x, int x_stride,
                  uint8_t *w, int w_stride,
                  fp16 *y, int y_stride)
{
    dim3 blockSize(1, 128);
    dim3 gridSize((B + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);
    kernel_mm8_seq<<<gridSize, blockSize>>>(
        B, N, M, cast(x), x_stride, w, w_stride,
        cast(y), y_stride);
}
