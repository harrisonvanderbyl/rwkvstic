#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#include <cuda_fp16.h>
#define MIN_VALUE (-1e38)







#define MM8_ONE_JSPLIT 16
#define MM8_ONE_TILE 1024



__global__ void kernel_mm8_one(
    const int N, const int M,
    const float *__restrict__ const x,
    const uint8_t *__restrict__ const w, const int w_stride,
    float *__restrict__ const y,
    const float *__restrict__ const r
    ){

    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
    const int j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

    if (k < M) {
        float y_local = 0;
        for (int j = j0; j < j1; ++j) {
            y_local += x[j] * (
                (w[j * w_stride + k] * r[j])
                
            );
        }
        atomicAdd(reinterpret_cast<float *>(&y[k]), *reinterpret_cast<float *>(&y_local));
    }
}



__global__ void kernel_mm8_three(
    const int N, const int M,
    const float *__restrict__ const x,
    const float *__restrict__ const x1,
    const float *__restrict__ const x2,

    const uint8_t *__restrict__ const w, const int w_stride,
    const uint8_t *__restrict__ const w1, const int w1_stride,
    const uint8_t *__restrict__ const w2, const int w2_stride,
    float *__restrict__ const y,
    float *__restrict__ const y1,
    float *__restrict__ const y2,
    const float *__restrict__ const r,
    const float *__restrict__ const r1,
    const float *__restrict__ const r2
    
    ){

    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
    const int j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

    if (k < M) {
        float y_local = 0;
        float y1_local = 0;
        float y2_local = 0;
        for (int j = j0; j < j1; ++j) {
            y_local += x[j] * (
                (w[j * w_stride + k] * r[j]));
            y1_local += x1[j] * (
                (w1[j * w1_stride + k] * r1[j]));
            y2_local += x2[j] * (
                (w2[j * w2_stride + k] * r2[j]));
           
        }
        atomicAdd(reinterpret_cast<float *>(&y[k]), *reinterpret_cast<float *>(&y_local));
        atomicAdd(reinterpret_cast<float *>(&y1[k]), *reinterpret_cast<float *>(&y1_local));
        atomicAdd(reinterpret_cast<float *>(&y2[k]), *reinterpret_cast<float *>(&y2_local));
    }
}
// generic T either float or fp16 or fp64

void cuda_mm8_three(int N, int M,
                    float *x,
                    float *x1,
                    float *x2,
                    uint8_t *w, int w_stride,
                    uint8_t *w1, int w1_stride,
                    uint8_t *w2, int w2_stride,
                    float *y,
                    float *y1,
                    float *y2,
                    float *r  ,
                    float *r1,
                    float *r2 
                ) {
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (M + blockSize.y - 1) / blockSize.y);
    kernel_mm8_three<<<gridSize, blockSize>>>(
        N, M, x, x1, x2, w, w_stride, w1, w1_stride, w2, w2_stride, y, y1, y2, r, r1, r2);
}
void cuda_mm8_one(int N, int M,
                  float *x,
                  uint8_t *w, int w_stride,
                  float *y,
                    float *r   
                ) {
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (M + blockSize.y - 1) / blockSize.y);
    kernel_mm8_one<<<gridSize, blockSize>>>(
        N, M, x, w, w_stride,y, r);
                }
