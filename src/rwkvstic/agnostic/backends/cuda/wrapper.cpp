#include <torch/extension.h>
#include "ATen/ATen.h"
#include <iostream>
#include <c10/cuda/CUDAGuard.h>
// generic T either float or fp16 or fp64


void cuda_mm8_three(int N, int M,
                    double *x,
                    double *x1,
                    double *x2,
                    uint8_t *w, int w_stride,
                    uint8_t *w1, int w1_stride,
                    uint8_t *w2, int w2_stride,
                    double *y,
                    double *y1,
                    double *y2,
                    double *r, double *r1, double *r2
                    );
void cuda_mm8_one(int N, int M,
                  double *x,
                  uint8_t *w, int w_stride,
                  double *y,
                    double *r
                    );
void cuda_wkv_forward(int B, int T, int C, double *w, double *u, double *k, double *v, double *y, double *aa, double *bb, double *pp);
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
                    float *r, float *r1, float *r2
                    );
void cuda_mm8_one(int N, int M,
                  float *x,
                  uint8_t *w, int w_stride,
                  float *y,
                    float *r
                    );
void wkv_forward(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &aa, torch::Tensor &bb, torch::Tensor &pp) {
    assert(w.scalar_type() == torch::kDouble);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));
    cuda_wkv_forward(B, T, C, w.data_ptr<double>(), u.data_ptr<double>(), k.data_ptr<double>(), v.data_ptr<double>(), y.data_ptr<double>(), aa.data_ptr<double>(), bb.data_ptr<double>(), pp.data_ptr<double>());
    
}

void mm8_one(int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &y,torch::Tensor &r) {
    assert(x.stride(0) == 1);
    assert(w.stride(1) == 1);
    assert(y.stride(0) == 1);
    assert(x.scalar_type() == y.scalar_type() && x.scalar_type() == r.scalar_type());
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    if( x.scalar_type()== torch::kDouble){
        cuda_mm8_one(
        N, M,
        x.data_ptr<double>(),
        w.data_ptr<uint8_t>(), w.stride(0),
        y.data_ptr<double>(),
        r.data_ptr<double>()
        );
    
    }else{
        cuda_mm8_one(
        N, M,
        x.data_ptr<float>(),
        w.data_ptr<uint8_t>(), w.stride(0),
        y.data_ptr<float>(),
        r.data_ptr<float>()
        );
    }

    
}

void mm8_three(int64_t N, int64_t M,
               torch::Tensor &x, torch::Tensor &x1, torch::Tensor &x2,
               torch::Tensor &w, torch::Tensor &w1, torch::Tensor &w2,
               torch::Tensor &y, torch::Tensor &y1, torch::Tensor &y2,
               torch::Tensor &r, torch::Tensor &r1, torch::Tensor &r2) {
    assert(x.stride(0) == 1);
    assert(x1.stride(0) == 1);
    assert(x2.stride(0) == 1);
    assert(w.stride(1) == 1);
    assert(w1.stride(1) == 1);
    assert(w2.stride(1) == 1);
    assert(y.stride(0) == 1);
    assert(y1.stride(0) == 1);
    assert(y2.stride(0) == 1);
    assert(x.scalar_type() == y.scalar_type() && x.scalar_type() == r.scalar_type());
    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    if(x.scalar_type() == torch::kDouble){
        cuda_mm8_three(
        N, M,
        x.data_ptr<double>(),
        x1.data_ptr<double>(),
        x2.data_ptr<double>(),
        w.data_ptr<uint8_t>(), w.stride(0),
        w1.data_ptr<uint8_t>(), w1.stride(0),
        w2.data_ptr<uint8_t>(), w2.stride(0),
        y.data_ptr<double>(),
        y1.data_ptr<double>(),
        y2.data_ptr<double>(),
        r.data_ptr<double>(), r1.data_ptr<double>(), r2.data_ptr<double>()
        );
    }
    else{
        cuda_mm8_three(
        N, M,
        x.data_ptr<float>(),
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        w.data_ptr<uint8_t>(), w.stride(0),
        w1.data_ptr<uint8_t>(), w1.stride(0),
        w2.data_ptr<uint8_t>(), w2.stride(0),
        y.data_ptr<float>(),
        y1.data_ptr<float>(),
        y2.data_ptr<float>(),
        r.data_ptr<float>(), r1.data_ptr<float>(), r2.data_ptr<float>()
        );}
    
   
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wkv_forward", &wkv_forward, "wkv forward");
    m.def("mm8_one", &mm8_one, "mm8 one");
    m.def("mm8_three", &mm8_three, "mm8 three");
}

TORCH_LIBRARY(rwkv, m) {
    m.def("wkv_forward", wkv_forward);
    m.def("mm8_one", mm8_one);
    m.def("mm8_three", mm8_three);
}
