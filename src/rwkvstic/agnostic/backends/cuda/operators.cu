#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#include <cuda_fp16.h>
#define MIN_VALUE (-1e38)
// typedef at::float fp16;
#define fp16 float
//#define DTYPE __half
#define DTYPE float
__global__ void kernel_wkv_forward(const int B, const int T, const int C,
                               const float *__restrict__ const _w, const float *__restrict__ const _u, const fp16 *__restrict__ const _k, const fp16 *__restrict__ const _v,
                               fp16 *__restrict__ const _y, float *__restrict__ const _aa, float *__restrict__ const _bb, float *__restrict__ const _pp) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int _b = idx / C;
    const int _c = idx % C;
    const int _offset = _b * T * C + _c;
    const int _state_offset = _b * C + _c;

    float u = _u[_c];
    float w = _w[_c];
    const fp16 *__restrict__ const k = _k + _offset;
    const fp16 *__restrict__ const v = _v + _offset;
    fp16 *__restrict__ const y = _y + _offset;

    float aa = _aa[_state_offset];
    float bb = _bb[_state_offset];
    float pp = _pp[_state_offset];
    for (int i = 0; i < T; i++) {
        const int ii = i * C;
        const float kk = float(k[ii]);
        const float vv = float(v[ii]);
        float ww = u + kk;
        float p = max(pp, ww);
        float e1 = exp(pp - p);
        float e2 = exp(ww - p);
        y[ii] = fp16((e1 * aa + e2 * vv) / (e1 * bb + e2));
        ww = w + pp;
        p = max(ww, kk);
        e1 = exp(ww - p);
        e2 = exp(kk - p);
        aa = e1 * aa + e2 * vv;
        bb = e1 * bb + e2;
        pp = p;
    }
    _aa[_state_offset] = aa;
    _bb[_state_offset] = bb;
    _pp[_state_offset] = pp;
}
void cuda_wkv_forward(int B, int T, int C, float *w, float *u, fp16 *k, fp16 *v, fp16 *y, float *aa, float *bb, float *pp) {
    dim3 threadsPerBlock( min(C, 32) );
    assert(B * C % threadsPerBlock.x == 0);
    dim3 numBlocks(B * C / threadsPerBlock.x);
    kernel_wkv_forward<<<numBlocks, threadsPerBlock>>>(B, T, C, w, u, k, v, y, aa, bb, pp);
}

DTYPE *cast(fp16 *ptr)
{
    return reinterpret_cast<DTYPE *>(ptr);
}

__global__ void kernel_mm8_seq(
    const int B, const int N, const int M,
    const fp16 *__restrict__ const x, const int x_stride,
    const uint8_t *__restrict__ const w, const int w_stride,
    fp16 *__restrict__ const y, const int y_stride,
    fp16 *__restrict__ const r

    )
{

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int k = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < B && k < M)
    {
        float y_local = 0;
        for (int j = 0; j < N; ++j)
        {
            y_local +=(x[i * x_stride + j]) * (w[k * w_stride + j] * (r[j]));
        }
        y[i * y_stride + k] = (y_local);
    }
}
void cuda_mm8_seq(int B, int N, int M,
                  fp16 *x, int x_stride,
                  uint8_t *w, int w_stride,
                  fp16 *y, int y_stride,
                    fp16 *r
                  )
{
    dim3 blockSize(1, 128);
    dim3 gridSize((B + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    kernel_mm8_seq<<<gridSize, blockSize>>>(
        B, N, M, (x), x_stride, w, w_stride,
        (y), y_stride,(r));
}

#define MM8_ONE_JSPLIT 24
#define MM8_ONE_TILE 1024

__global__ void kernel_mm8_one(
    const int N, const int M,
    const fp16 *__restrict__ const x,
    const uint8_t *__restrict__ const w, const int w_stride,
    fp16 *__restrict__ const y,
    const fp16 *__restrict__ const r
    ){

    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
    const int j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

    if (k < M) {
        fp16 y_local = fp16(0);
        for (int j = j0; j < j1; ++j) {
            y_local += x[j] * (
                (w[j * w_stride + k] * r[j])
                
            );
        }
        atomicAdd(reinterpret_cast<DTYPE *>(&y[k]), *reinterpret_cast<DTYPE *>(&y_local));
    }
}
void cuda_mm8_one(int N, int M,
                  fp16 *x,
                  uint8_t *w, int w_stride,
                  fp16 *y,
                    fp16 *r   
                ) {
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (M + blockSize.y - 1) / blockSize.y);
    kernel_mm8_one<<<gridSize, blockSize>>>(
        N, M, x, w, w_stride,y, r);
}

__global__ void kernel_mm8_three(
    const int N, const int M,
    const fp16 *__restrict__ const x,
    const fp16 *__restrict__ const x1,
    const fp16 *__restrict__ const x2,
    
    const uint8_t *__restrict__ const w, const int w_stride,
    const uint8_t *__restrict__ const w1, const int w1_stride,
    const uint8_t *__restrict__ const w2, const int w2_stride,
    fp16 *__restrict__ const y,
    fp16 *__restrict__ const y1,
    fp16 *__restrict__ const y2,
    const fp16 *__restrict__ const r,
    const fp16 *__restrict__ const r1,
    const fp16 *__restrict__ const r2
    
    ){

    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
    const int j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

    if (k < M) {
        fp16 y_local = at::Half(0);
        fp16 y1_local = at::Half(0);
        fp16 y2_local = at::Half(0);
        for (int j = j0; j < j1; ++j) {
            y_local += x[j] * (
                (w[j * w_stride + k] * r[j]));
            y1_local += x1[j] * (
                (w1[j * w1_stride + k] * r1[j]));
            y2_local += x2[j] * (
                (w2[j * w2_stride + k] * r2[j]));
           
        }
        atomicAdd(reinterpret_cast<DTYPE *>(&y[k]), *reinterpret_cast<DTYPE *>(&y_local));
        atomicAdd(reinterpret_cast<DTYPE *>(&y1[k]), *reinterpret_cast<DTYPE *>(&y1_local));
        atomicAdd(reinterpret_cast<DTYPE *>(&y2[k]), *reinterpret_cast<DTYPE *>(&y2_local));
    }
}
void cuda_mm8_three(int N, int M,
                    fp16 *x,
                    fp16 *x1,
                    fp16 *x2,
                    uint8_t *w, int w_stride,
                    uint8_t *w1, int w1_stride,
                    uint8_t *w2, int w2_stride,
                    fp16 *y,
                    fp16 *y1,
                    fp16 *y2,
                    fp16 *r  ,
                    fp16 *r1,
                    fp16 *r2 
                ) {
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (M + blockSize.y - 1) / blockSize.y);
    kernel_mm8_three<<<gridSize, blockSize>>>(
        N, M, x, x1, x2, w, w_stride, w1, w1_stride, w2, w2_stride, y, y1, y2, r, r1, r2);
}

