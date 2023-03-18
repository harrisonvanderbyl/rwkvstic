#include <torch/extension.h>
#include "ATen/ATen.h"
#include <iostream>
// typedef at::half fp16;
// typedef float fp16;
#define fp16 float




void cuda_mm8_seq(int B, int N, int M,
                  fp16 *x, int x_stride,
                  uint8_t *w, int w_stride,

                  fp16 *y, int y_stride
                  ,
                  fp16 *r
                  );
void cuda_mm8_one(int N, int M,
                  fp16 *x,
                  uint8_t *w, int w_stride,
                  fp16 *y,
                    fp16 *r
                    );
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
                    fp16 *r, fp16 *r1, fp16 *r2
                    );
void mm8_seq(int64_t B, int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &y,torch::Tensor &r)
{
    // assert(x.stride(1) == 1);
    // assert(w.stride(0) == 1);

    // assert(y.stride(1) == 1);
    cuda_mm8_seq(
        B, N, M,
        x.data_ptr<fp16>(), x.stride(0),
        w.data_ptr<uint8_t>(), w.stride(0),
        y.data_ptr<fp16>(), y.stride(0),
        r.data_ptr<fp16>()
        );
}
void cuda_wkv_forward(int B, int T, int C, float *w, float *u, fp16 *k, fp16 *v, fp16 *y, float *aa, float *bb, float *pp);
void wkv_forward(int64_t B, int64_t T, int64_t C, torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &aa, torch::Tensor &bb, torch::Tensor &pp) {
    cuda_wkv_forward(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<fp16>(), v.data_ptr<fp16>(), y.data_ptr<fp16>(), aa.data_ptr<float>(), bb.data_ptr<float>(), pp.data_ptr<float>());
}

void mm8_one(int64_t N, int64_t M,
             torch::Tensor &x, torch::Tensor &w,
             torch::Tensor &y,torch::Tensor &r) {
    assert(x.stride(0) == 1);
    assert(w.stride(1) == 1);
    assert(y.stride(0) == 1);
    cuda_mm8_one(
        N, M,
        x.data_ptr<fp16>(),
        w.data_ptr<uint8_t>(), w.stride(0),
        y.data_ptr<fp16>(),
        r.data_ptr<fp16>()
        );
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
    cuda_mm8_three(
        N, M,
        x.data_ptr<fp16>(),
        x1.data_ptr<fp16>(),
        x2.data_ptr<fp16>(),
        w.data_ptr<uint8_t>(), w.stride(0),
        w1.data_ptr<uint8_t>(), w1.stride(0),
        w2.data_ptr<uint8_t>(), w2.stride(0),
        y.data_ptr<fp16>(),
        y1.data_ptr<fp16>(),
        y2.data_ptr<fp16>(),
        r.data_ptr<fp16>(), r1.data_ptr<fp16>(), r2.data_ptr<fp16>()
        );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wkv_forward", &wkv_forward, "wkv forward");
    m.def("mm8_seq", &mm8_seq, "mm8 seq");
    m.def("mm8_one", &mm8_one, "mm8 one");
    m.def("mm8_three", &mm8_three, "mm8 three");
}

TORCH_LIBRARY(rwkv, m) {
    m.def("wkv_forward", wkv_forward);
    m.def("mm8_seq", mm8_seq);
    m.def("mm8_one", mm8_one);
    m.def("mm8_three", mm8_three);
}
