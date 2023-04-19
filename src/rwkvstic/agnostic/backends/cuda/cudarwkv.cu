
// xee2 = x - torch.mean(x, dim=1, keepdim=True)

// x2 = torch.sqrt(torch.mean(xee2*xee2, dim=1, keepdim=True) +
//                 1e-5)

// return self.weight*(xee2/x2) + self.bias
#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#include <cuda_fp16.h>
#define MM8_ONE_JSPLIT 16
#define MM8_ONE_TILE 1024
#define EMBSPLIT 64
#define EMBBLOCK 8

__global__ void cuda_layernormMean(int64_t n_emb, const double *__restrict__ const x, double *__restrict__ const out)
{

    double xmean = 0;
    for (int64_t i = 0; i < n_emb; i++)
    {
        xmean += x[i] / n_emb;
    }

    double x2 = 0;
    for (int64_t i = 0; i < n_emb; i++)
    {
        x2 += (x[i] - xmean) * (x[i] - xmean) / n_emb;
    }
    x2 = sqrt(x2 + 1e-5);
    out[0] = xmean;
    out[1] = x2;
}
__global__ void cuda_layernorm(int64_t n_emb, const double *__restrict__ const x, const double *__restrict__ const weight, int64_t offset, double *__restrict__ const in, double *__restrict__ const out)
    {
    double xmean = in[0];
    double x2 = in[1];

    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
        if(i < n_emb){
            out[i] = weight[n_emb * offset + i] * ((x[i] - xmean) / x2) + weight[n_emb * (offset + 1) + i];
    }}
}
__global__ void kernel_mm8_threec(
    const int64_t N,
    const double *__restrict__ const xy,
    double *__restrict__ const statexy,
    double *__restrict__ const mixk,
    double *__restrict__ const mixv,
    double *__restrict__ const mixr,
    const uint8_t *__restrict__ const w,
    const uint8_t *__restrict__ const w1,
    const uint8_t *__restrict__ const w2,
    const float *__restrict__ const r,
    const float *__restrict__ const r1,
    const float *__restrict__ const r2,
    const float *__restrict__ const o1,
    const float *__restrict__ const o2,
    const float *__restrict__ const o3,
    float *__restrict__ const y,
    float *__restrict__ const y1,
    float *__restrict__ const y2,
    int64_t offset)
{

    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
    const int j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

    if (k < N)
    {
        float y_local = 0;
        float y1_local = 0;
        float y2_local = 0;
        for (int j = j0; j < j1; ++j)
        {
            y_local += float(xy[j] * mixk[j + offset * N] + statexy[j + offset * N] * (1.0 - mixk[j + offset * N])) * ((w[j * N + k + offset * N * N] * r[j + offset * N]) + o1[j + offset * N]);
            y1_local += float(xy[j] * mixv[j + offset * N] + statexy[j + offset * N] * (1.0 - mixv[j + offset * N])) * ((w1[j * N + k + offset * N * N] * r1[j + offset * N]) + o2[j + offset * N]);
            y2_local += float(xy[j] * mixr[j + offset * N] + statexy[j + offset * N] * (1.0 - mixr[j + offset * N])) * ((w2[j * N + k + offset * N * N] * r2[j + offset * N]) + o3[j + offset * N]);
        }
        atomicAdd(reinterpret_cast<float *>(&y[k]), *reinterpret_cast<float *>(&y_local));
        atomicAdd(reinterpret_cast<float *>(&y1[k]), *reinterpret_cast<float *>(&y1_local));
        atomicAdd(reinterpret_cast<float *>(&y2[k]), *reinterpret_cast<float *>(&y2_local));
    }
}

// generic T either float or fp16 or fp64

void cuda_mm8_threec(int64_t N,
                     double *xy,
                     double *statexy,
                     double *mixk,
                     double *mixv,
                     double *mixr,
                     uint8_t *w,
                     uint8_t *w1,
                     uint8_t *w2,
                     float *r,
                     float *r1,
                     float *r2,
                     float *o1,
                     float *o2,
                     float *o3,
                     float *y,
                     float *y1,
                     float *y2,
                     int64_t offset = 0

)
{
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (N + blockSize.y - 1) / blockSize.y);
    kernel_mm8_threec<<<gridSize, blockSize>>>(
        N,
        xy,
        statexy,
        mixk,
        mixv,
        mixr,
        w,
        w1,
        w2,
        r,
        r1,
        r2,
        o1,
        o2,
        o3,
        y,
        y1,
        y2,
        offset);
}

__global__ void setx(
    const int emb,
    const float *__restrict__ const a,
    double *__restrict__ const b,
    int64_t offset = 0)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        b[i + offset * emb] = double(a[i]);
    }}
}
__global__ void setx(
    const int emb,
    double *__restrict__ const a,
    double *__restrict__ const b,
    int64_t offset = 0)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        b[i + offset * emb] = double(a[i]);
    }}
}
__global__ void setx(
    const int emb,
    double *__restrict__ const a,
    float *__restrict__ const b,
    int64_t offset = 0)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        b[i + offset * emb] = float(a[i]);
    }}
}
__global__ void cuda_memset(
    const int emb,
    double *__restrict__ const a,
    double b)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        a[i] = b;
    }}
}

__global__ void cuda_memset(
    const int emb,
    float *__restrict__ const a,
    float b)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        a[i] = b;
    }}
}
__global__ void cuda_relusquared(
    const int emb,
    float *__restrict__ const a)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        a[i] = a[i] * (a[i] > 0);
        a[i] = a[i] * a[i];
    }}
}

__global__ void sigmoid(
    const int emb,
    float *__restrict__ const a,
    double *__restrict__ const out)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        out[i] = 1.0 / (1.0 + exp(-double(a[i])));
    }}
}
__global__ void kernel_wkvc_forward(const int C,
                                    const double *__restrict__ const w, const double *__restrict__ const u, const float *__restrict__ const k, const float *__restrict__ const v,
                                    const float *__restrict__ const r, double *__restrict__ const y, double *__restrict__ const _aa, double *__restrict__ const _bb, double *__restrict__ const _pp,
                                    int64_t offset)
{

    int i = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int ii = i + threadIdx.x*EMBBLOCK + c;
    
    if(ii < C){
        double aa = _aa[ii + C * offset];
        double bb = _bb[ii + C * offset];
        double pp = _pp[ii + C * offset];

        const double vv = v[ii];
        const double wr1 = aa + exp(u[ii + C * offset] + w[ii + C * offset] + k[ii]) * vv;
        const double wr2 = bb + exp(u[ii + C * offset] + w[ii + C * offset] + k[ii]);
        y[ii] = (wr1) / wr2;
        y[ii] = (1.0 / (1.0 + exp(-r[ii]))) * y[ii];
        aa = (aa + exp(k[ii]) * vv) * exp(w[ii + C * offset]);
        bb = (bb + exp(k[ii])) * exp(w[ii + C * offset]);
        _aa[ii + C * offset] = aa;
        _bb[ii + C * offset] = bb;
        _pp[ii + C * offset] = pp;
    }}
}

void cuda_wkvc_forward(int B, double *w, double *u, float *k, float *v, float *r, double *y, double *aa, double *bb, double *pp, int64_t offset)
{

    kernel_wkvc_forward<<<(B+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(B, w, u, k, v, r, y, aa, bb, pp, offset);
}
__global__ void kernelc_mm8_one(
    const int N, const int M,
    const double *__restrict__ const x,
    const uint8_t *__restrict__ const w, const int w_stride,
    float *__restrict__ const y,
    const float *__restrict__ const r,
    const float *__restrict__ const o,
    const int64_t offset)
{

    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
    const int j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

    if (k < M)
    {
        float y_local = 0;
        for (int j = j0; j < j1; ++j)
        {
            y_local += float(x[j]) * ((w[j * w_stride + k + offset * N * M] * r[j + offset * N] + o[j + offset * N]));
        }
        atomicAdd(reinterpret_cast<float *>(&y[k]), *reinterpret_cast<float *>(&y_local));
    }
}

void cudac_mm8_one(int N, int M,
                   double *x,
                   uint8_t *w, int w_stride,
                   float *y,
                   float *r,
                   float *o,
                   uint64_t offset)
{
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (M + blockSize.y - 1) / blockSize.y);
    kernelc_mm8_one<<<gridSize, blockSize>>>(
        N, M, x, w, w_stride, y, r, o, offset);
}

__global__ void kernelc_mm8_one(
    const int N, const int M,
    const float *__restrict__ const x,
    const uint8_t *__restrict__ const w, const int w_stride,
    float *__restrict__ const y,
    const float *__restrict__ const r,
    const float *__restrict__ const o,
    const int64_t offset)
{

    const int k = blockIdx.y * blockDim.y + threadIdx.y;
    const int j0 = min(N, blockIdx.x * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));
    const int j1 = min(N, (blockIdx.x + 1) * ((N + MM8_ONE_JSPLIT - 1) / MM8_ONE_JSPLIT));

    if (k < M)
    {
        float y_local = 0;
        for (int j = j0; j < j1; ++j)
        {
            y_local += float(x[j]) * ((w[j * w_stride + k + offset * N * M] * r[j + offset * N] + o[j + offset * N]));
        }
        atomicAdd(reinterpret_cast<float *>(&y[k]), *reinterpret_cast<float *>(&y_local));
    }
}

void cudac_mm8_one(int N, int M,
                   float *x,
                   uint8_t *w, int w_stride,
                   float *y,
                   float *r,
                   float *o,
                   uint64_t offset)
{
    dim3 blockSize(1, MM8_ONE_TILE);
    dim3 gridSize(MM8_ONE_JSPLIT, (M + blockSize.y - 1) / blockSize.y);
    kernelc_mm8_one<<<gridSize, blockSize>>>(
        N, M, x, w, w_stride, y, r, o, offset);
}
__global__ void addx(
    const int emb,
    double *__restrict__ const a,
    double *__restrict__ const b)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        b[i] += double(a[i]);
    }}
}

__global__ void mixffn(
    const int emb,
    double *__restrict__ const rc,
    double *__restrict__ const ddd,
    double *__restrict__ const mixk,
    double *__restrict__ const mixr,
    double *__restrict__ const outk,
    double *__restrict__ const outr, int64_t offset

)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        outk[i] = mixk[i + offset * emb] * rc[i] + (1.0 - mixk[i + offset * emb]) * ddd[i + offset * emb];
        outr[i] = mixr[i + offset * emb] * rc[i] + (1.0 - mixr[i + offset * emb]) * ddd[i + offset * emb];
    }}
}

__global__ void blockout(
    const int emb,
    double *__restrict__ const x,
    float *__restrict__ const rcm,
    double *__restrict__ const ddd)
{
    int ii = blockIdx.x*(blockDim.x*EMBBLOCK);
    for(int c = 0; c < EMBBLOCK; c++){
        int i = ii + threadIdx.x*EMBBLOCK + c;
    
    if(i < emb){
        x[i] = x[i] + rcm[i] * ddd[i];
    }}
}

void cuda_rwkv(int64_t n_layers, int64_t n_emb, int64_t token, double *x,
               double *embed, double *layernorms,
               double *statexy, double *stateaa, double *statebb, double *statepp, double *statedd,
               double *buffer1, float *buffer2, float *buffer3, float *buffer4,
               double *mixk, double *mixv, double *mixr,
               uint8_t *km, uint8_t *vm, uint8_t *rm,
               float *kr, float *vr, float *rr,
               float *o1, float *o2, float *o3,
               uint8_t *attout, float *attoutr, float *attouto,
               double *ffnmixk, double *ffnmixv,
               uint8_t *ffnk, uint8_t *ffnv, uint8_t *ffnr,
               float *ffnkr, float *ffnvr, float *ffnrr,
               float *ffnko, float *ffnvo, float *ffnro,
               double *ffnkbuffer, double *ffnvbuffer, float *ffnrbuffer,
               double *decay, double *bonus,
               uint8_t *head, float *headr, float *heado)
{
    cuda_layernormMean<<<1, 1>>>(n_emb, embed, ffnkbuffer);
    cuda_layernorm<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, embed, layernorms, 0,ffnkbuffer, x);

    for (int64_t i = 0; i < n_layers; i++)
    {
        // xy = ln(x)
        // kmix, vmix, rmix = mix(xy, statexy[n_emb*y], mixkvr)
        // k, v, r = matmul(kmix, km), matmul(vmix, vm), matmul(rmix, rm)
        cuda_layernormMean<<<1, 1>>>(n_emb, x, ffnkbuffer);
        cuda_layernorm<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, x, layernorms, 4 * (i) + 2,ffnkbuffer, buffer1);
        // buffers to 0
        cuda_memset<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer2, 0);
        cuda_memset<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer3, 0);
        cuda_memset<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer4, 0);
        cuda_mm8_threec(n_emb, buffer1, statexy, mixk, mixv, mixr, km, vm, rm, kr, vr, rr, o1, o2, o3, buffer2, buffer3, buffer4, i);
        setx<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer1, statexy, i);
        // buffer2, 3, 4 = k,v,r

        cuda_wkvc_forward(n_emb, decay, bonus, buffer2, buffer3, buffer4, buffer1, stateaa, statebb, statepp, i);

        // buffer1 = rwkv
        setx<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, x, buffer2);
        cudac_mm8_one(n_emb, n_emb, buffer1, attout, n_emb, buffer2, attoutr, attouto, i);
        // buffer2 = attout(rwkv) + x
        setx<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer2, x);
        cuda_layernormMean<<<1, 1>>>(n_emb, x, ffnkbuffer);
        cuda_layernorm<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, x, layernorms, 4 * (i + 1),ffnkbuffer, buffer1);
        // buffer1 = ln(x)
        // ffnmixk, ffnmixv = mix(buffer1, statedd[n_emb*y], ffnmixkvr)
        mixffn<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer1, statedd, ffnmixk, ffnmixv, ffnkbuffer, ffnvbuffer, i);
        setx<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer1, statedd, i);
        // ffnkbuffer, ffnvbuffer = mixk, mixv
        cuda_memset<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer2, 0);
        cudac_mm8_one(n_emb, n_emb, ffnvbuffer, ffnr, n_emb, buffer2, ffnrr, ffnro, i);
        sigmoid<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer2, ffnvbuffer);
        // ffnvbuffer = sigmoid(ffnrbuffer @ ffnr)
        cuda_memset<<<(n_emb*4+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb * 4, ffnrbuffer, 0);
        cudac_mm8_one(n_emb, n_emb * 4, ffnkbuffer, ffnk, n_emb * 4, ffnrbuffer, ffnkr, ffnko, i);
        // ffnrbuffer = ffnkbuffer @ ffnk
        cuda_relusquared<<<(n_emb*4+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb * 4, ffnrbuffer);
        cuda_memset<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, buffer3, 0);
        cudac_mm8_one(n_emb * 4, n_emb, ffnrbuffer, ffnv, n_emb, buffer3, ffnvr, ffnvo, i);
        // buffer3 = ffnrbuffer @ ffnv
        blockout<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, x, buffer3, ffnvbuffer);

        // cuda_layernorm<<<1,1>>>(n_emb, x, layernorms,4*(i)+4, buffer1);
        // setx<<<1,1>>>(n_emb, buffer1, x);
    }

    cuda_layernormMean<<<1, 1>>>(n_emb, x, ffnkbuffer);
    cuda_layernorm<<<(n_emb+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(n_emb, x, layernorms, 4 * (n_layers) + 2,ffnkbuffer, buffer1);
    cuda_memset<<<(50277+EMBSPLIT-1)/EMBSPLIT, EMBSPLIT/EMBBLOCK>>>(50277, buffer2, 0);
    cudac_mm8_one(n_emb, 50277, buffer1, head, 50277, buffer2, headr, heado, 0);
}