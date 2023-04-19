#include <torch/extension.h>
#include "ATen/ATen.h"
#include <iostream>
#include <c10/cuda/CUDAGuard.h>
// generic T either float or fp16 or fp64

void cuda_rwkv(int64_t n_layers, int64_t n_emb, int64_t token, double* x, 
    double* embed, double* layernorms, 
    double* statexy, double* stateaa, double* statebb, double* statepp, double* statedd,
    double* buffer1, float* buffer2, float* buffer3, float* buffer4,
    double* mixk, double* mixv, double* mixr,
    uint8_t* km, uint8_t* vm, uint8_t* rm,
    float* kr, float* vr, float* rr,
    float* o1, float* o2, float* o3,
    uint8_t* attout, float* attoutr, float* attouto,
    double* ffnmixk, double* ffnmixv,
    uint8_t* ffnk, uint8_t* ffnv, uint8_t* ffnr, 
    float* ffnkr, float* ffnvr, float* ffnrr, 
    float* ffnko, float* ffnvo, float* ffnro,
    double* ffnkbuffer, double* ffnvbuffer, float* ffnrbuffer,
    double* decay, double* bonus,
    uint8_t* head, float* headr, float* heado
    );
void cuda_rwkv_wrapper(int64_t n_layers, int64_t n_emb, int64_t token, torch::Tensor &x, 
    torch::Tensor &embed, torch::Tensor &layernorms, 
    torch::Tensor &statexy, torch::Tensor &stateaa, torch::Tensor &statebb, torch::Tensor &statepp, torch::Tensor &statedd,
    torch::Tensor &buffer1, torch::Tensor &buffer2, torch::Tensor &buffer3, torch::Tensor &buffer4,
    torch::Tensor &mixk, torch::Tensor &mixv, torch::Tensor &mixr,
    torch::Tensor &km, torch::Tensor &vm, torch::Tensor &rm,
    torch::Tensor &kr, torch::Tensor &vr, torch::Tensor &rr,
    torch::Tensor &o1, torch::Tensor &o2, torch::Tensor &o3,
    torch::Tensor &attout, torch::Tensor &attoutr, torch::Tensor &attouto,
    torch::Tensor &ffnmixk, torch::Tensor &ffnmixv,
    torch::Tensor &ffnk, torch::Tensor &ffnv, torch::Tensor &ffnr, 
    torch::Tensor &ffnkr, torch::Tensor &ffnvr, torch::Tensor &ffnrr, 
    torch::Tensor &ffnko, torch::Tensor &ffnvo, torch::Tensor &ffnro, 
    torch::Tensor &ffnkbuffer, torch::Tensor &ffnvbuffer, torch::Tensor &ffnrbuffer,

    torch::Tensor &decay, torch::Tensor &bonus,
    torch::Tensor &head, torch::Tensor &headr, torch::Tensor &heado
    ) {
    assert(x.scalar_type() == torch::kDouble);
    assert(embed.scalar_type() == torch::kDouble);
    assert(layernorms.scalar_type() == torch::kDouble);
    assert(statexy.scalar_type() == torch::kDouble);
    assert(stateaa.scalar_type() == torch::kDouble);
    assert(statebb.scalar_type() == torch::kDouble);
    assert(statepp.scalar_type() == torch::kDouble);
    assert(statedd.scalar_type() == torch::kDouble);
    assert(buffer1.scalar_type() == torch::kDouble);
    assert(buffer2.scalar_type() == torch::kFloat);
    assert(buffer3.scalar_type() == torch::kFloat);
    assert(buffer4.scalar_type() == torch::kFloat);
    assert(mixk.scalar_type() == torch::kDouble);
    assert(mixv.scalar_type() == torch::kDouble);
    assert(mixr.scalar_type() == torch::kDouble);
    assert(km.scalar_type() == torch::kByte);
    assert(vm.scalar_type() == torch::kByte);
    assert(rm.scalar_type() == torch::kByte);
    assert(kr.scalar_type() == torch::kFloat);
    assert(vr.scalar_type() == torch::kFloat);
    assert(rr.scalar_type() == torch::kFloat);
    assert(decay.scalar_type() == torch::kDouble);
    assert(bonus.scalar_type() == torch::kDouble);
    const at::cuda::OptionalCUDAGuard device_guard(device_of (x));

    cuda_rwkv(n_layers, n_emb, token, x.data_ptr<double>(),
        embed.data_ptr<double>(), layernorms.data_ptr<double>(),
        statexy.data_ptr<double>(), stateaa.data_ptr<double>(), statebb.data_ptr<double>(), statepp.data_ptr<double>(), statedd.data_ptr<double>(),
        buffer1.data_ptr<double>(), buffer2.data_ptr<float>(), buffer3.data_ptr<float>(), buffer4.data_ptr<float>(),
        mixk.data_ptr<double>(), mixv.data_ptr<double>(), mixr.data_ptr<double>(),
        km.data_ptr<uint8_t>(), vm.data_ptr<uint8_t>(), rm.data_ptr<uint8_t>(),
        kr.data_ptr<float>(), vr.data_ptr<float>(), rr.data_ptr<float>(),
        o1.data_ptr<float>(), o2.data_ptr<float>(), o3.data_ptr<float>(),
        attout.data_ptr<uint8_t>(), attoutr.data_ptr<float>(), attouto.data_ptr<float>(),
        ffnmixk.data_ptr<double>(), ffnmixv.data_ptr<double>(),
        ffnk.data_ptr<uint8_t>(), ffnv.data_ptr<uint8_t>(), ffnr.data_ptr<uint8_t>(), 
        ffnkr.data_ptr<float>(), ffnvr.data_ptr<float>(), ffnrr.data_ptr<float>(), 
        ffnko.data_ptr<float>(), ffnvo.data_ptr<float>(), ffnro.data_ptr<float>(), 
        ffnkbuffer.data_ptr<double>(), ffnvbuffer.data_ptr<double>(), ffnrbuffer.data_ptr<float>(),
        decay.data_ptr<double>(), bonus.data_ptr<double>(),
        head.data_ptr<uint8_t>(), headr.data_ptr<float>(), heado.data_ptr<float>()
        );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rwkvc", &cuda_rwkv_wrapper, "rwkvc");
}

TORCH_LIBRARY(rwkv, m) {
    m.def("rwkvc", cuda_rwkv_wrapper);
}
