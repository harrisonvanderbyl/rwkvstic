#include <torch/extension.h>
#include "ATen/ATen.h"
#include <iostream>
// generic T either float or fp16 or fp64




void save(int64_t n_layers, int64_t n_emb, torch::Tensor &x, 
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
    assert(embed.scalar_type() == torch::kFloat);
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
    assert(o1.scalar_type() == torch::kFloat);
    assert(o2.scalar_type() == torch::kFloat);
    assert(o3.scalar_type() == torch::kFloat);
    assert(attout.scalar_type() == torch::kByte);
    assert(attoutr.scalar_type() == torch::kFloat);
    assert(attouto.scalar_type() == torch::kFloat);
    assert(ffnmixk.scalar_type() == torch::kDouble);
    assert(ffnmixv.scalar_type() == torch::kDouble);
    assert(ffnk.scalar_type() == torch::kByte);
    assert(ffnv.scalar_type() == torch::kByte);
    assert(ffnr.scalar_type() == torch::kByte);
    assert(ffnkr.scalar_type() == torch::kFloat);
    assert(ffnvr.scalar_type() == torch::kFloat);
    assert(ffnrr.scalar_type() == torch::kFloat);
    assert(ffnko.scalar_type() == torch::kFloat);
    assert(ffnvo.scalar_type() == torch::kFloat);
    assert(ffnro.scalar_type() == torch::kFloat);

    assert(ffnkbuffer.scalar_type() == torch::kDouble);
    assert(ffnvbuffer.scalar_type() == torch::kDouble);
    assert(ffnrbuffer.scalar_type() == torch::kFloat);

    assert(decay.scalar_type() == torch::kDouble);
    assert(bonus.scalar_type() == torch::kDouble);

    assert(head.scalar_type() == torch::kByte);
    assert(headr.scalar_type() == torch::kFloat);
    assert(heado.scalar_type() == torch::kFloat);

    // save each tensor consecutively to a single bin file, prepending with the n_layers and n_emb

    std::ofstream binfile;
    binfile.open("rwkv.bin", std::ios::out | std::ios::binary);
    binfile.write((char*)&n_layers, sizeof(int64_t));
    binfile.write((char*)&n_emb, sizeof(int64_t));

    binfile.write((char*)x.data_ptr<double>(), x.numel() * sizeof(double));
    binfile.write((char*)embed.data_ptr<float>(), embed.numel() * sizeof(float));
    binfile.write((char*)layernorms.data_ptr<double>(), layernorms.numel() * sizeof(double));
    binfile.write((char*)statexy.data_ptr<double>(), statexy.numel() * sizeof(double));
    binfile.write((char*)stateaa.data_ptr<double>(), stateaa.numel() * sizeof(double));
    binfile.write((char*)statebb.data_ptr<double>(), statebb.numel() * sizeof(double));
    binfile.write((char*)statepp.data_ptr<double>(), statepp.numel() * sizeof(double));
    binfile.write((char*)statedd.data_ptr<double>(), statedd.numel() * sizeof(double));
    binfile.write((char*)buffer1.data_ptr<double>(), buffer1.numel() * sizeof(double));
    binfile.write((char*)buffer2.data_ptr<float>(), buffer2.numel() * sizeof(float));
    binfile.write((char*)buffer3.data_ptr<float>(), buffer3.numel() * sizeof(float));
    binfile.write((char*)buffer4.data_ptr<float>(), buffer4.numel() * sizeof(float));
    binfile.write((char*)mixk.data_ptr<double>(), mixk.numel() * sizeof(double));
    binfile.write((char*)mixv.data_ptr<double>(), mixv.numel() * sizeof(double));
    binfile.write((char*)mixr.data_ptr<double>(), mixr.numel() * sizeof(double));
    binfile.write((char*)km.data_ptr<uint8_t>(), km.numel() * sizeof(uint8_t));
    binfile.write((char*)vm.data_ptr<uint8_t>(), vm.numel() * sizeof(uint8_t));
    binfile.write((char*)rm.data_ptr<uint8_t>(), rm.numel() * sizeof(uint8_t));
    binfile.write((char*)kr.data_ptr<float>(), kr.numel() * sizeof(float));
    binfile.write((char*)vr.data_ptr<float>(), vr.numel() * sizeof(float));
    binfile.write((char*)rr.data_ptr<float>(), rr.numel() * sizeof(float));
    binfile.write((char*)o1.data_ptr<float>(), o1.numel() * sizeof(float));
    binfile.write((char*)o2.data_ptr<float>(), o2.numel() * sizeof(float));
    binfile.write((char*)o3.data_ptr<float>(), o3.numel() * sizeof(float));
    binfile.write((char*)attout.data_ptr<uint8_t>(), attout.numel() * sizeof(uint8_t));
    binfile.write((char*)attoutr.data_ptr<float>(), attoutr.numel() * sizeof(float));
    binfile.write((char*)attouto.data_ptr<float>(), attouto.numel() * sizeof(float));
    binfile.write((char*)ffnmixk.data_ptr<double>(), ffnmixk.numel() * sizeof(double));
    binfile.write((char*)ffnmixv.data_ptr<double>(), ffnmixv.numel() * sizeof(double));
    binfile.write((char*)ffnk.data_ptr<uint8_t>(), ffnk.numel() * sizeof(uint8_t));
    binfile.write((char*)ffnv.data_ptr<uint8_t>(), ffnv.numel() * sizeof(uint8_t));
    binfile.write((char*)ffnr.data_ptr<uint8_t>(), ffnr.numel() * sizeof(uint8_t));
    binfile.write((char*)ffnkr.data_ptr<float>(), ffnkr.numel() * sizeof(float));
    binfile.write((char*)ffnvr.data_ptr<float>(), ffnvr.numel() * sizeof(float));
    binfile.write((char*)ffnrr.data_ptr<float>(), ffnrr.numel() * sizeof(float));
    binfile.write((char*)ffnko.data_ptr<float>(), ffnko.numel() * sizeof(float));
    binfile.write((char*)ffnvo.data_ptr<float>(), ffnvo.numel() * sizeof(float));
    binfile.write((char*)ffnro.data_ptr<float>(), ffnro.numel() * sizeof(float));
    binfile.write((char*)ffnkbuffer.data_ptr<double>(), ffnkbuffer.numel() * sizeof(double));
    binfile.write((char*)ffnvbuffer.data_ptr<double>(), ffnvbuffer.numel() * sizeof(double));
    binfile.write((char*)ffnrbuffer.data_ptr<float>(), ffnrbuffer.numel() * sizeof(float));
    binfile.write((char*)decay.data_ptr<double>(), decay.numel() * sizeof(double));
    binfile.write((char*)bonus.data_ptr<double>(), bonus.numel() * sizeof(double));
    binfile.write((char*)head.data_ptr<uint8_t>(), head.numel() * sizeof(uint8_t));
    binfile.write((char*)headr.data_ptr<float>(), headr.numel() * sizeof(float));
    binfile.write((char*)heado.data_ptr<float>(), heado.numel() * sizeof(float));

    binfile.close();

    }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("save", &save, "save");
}

TORCH_LIBRARY(rwkv, m) {
    m.def("save", save);
}
