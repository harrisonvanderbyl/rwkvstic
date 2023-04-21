#include <torch/extension.h>
#include "ATen/ATen.h"
#include <iostream>
#include <c10/cuda/CUDAGuard.h>
// generic T either float or fp16 or fp64
int** ptrs = new int*[46];
int num_layers = 0;
int num_embed = 0;

#include "structure.h"
// #define getsize[x] getSize(x, num_layers, num_embed)

void cuda_rwkv(int64_t n_layers, int64_t n_emb, int64_t token, double* x, 
    float* embed, double* layernorms, 
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

void cuda_rwkv_wrapper(int64_t token) {
    // print n_layers, n_embed
    // std::cout << "n_layers: " << num_layers << std::endl;
    // std::cout << "n_embed: " << num_embed << std::endl;
    // print by copying to cout
    // std::cout << "x: " << ptrs[x][0] << ":" << ptrs[x][num_embed-1] << std::endl;
    cuda_rwkv(num_layers, num_embed, token, (double*)(ptrs[X]),
        (float*)(ptrs[EMBED]), (double*)(ptrs[LAYERNORMS]),
        (double*)(ptrs[STATEXY]), (double*)(ptrs[STATEAA]), (double*)(ptrs[STATEBB]), (double*)(ptrs[STATEPP]), (double*)(ptrs[STATEDD]),
        (double*)(ptrs[BUFFER1]), (float*)(ptrs[BUFFER2]), (float*)(ptrs[BUFFER3]), (float*)(ptrs[BUFFER4]),
        (double*)(ptrs[MIXK]), (double*)(ptrs[MIXV]), (double*)(ptrs[MIXR]),
        (uint8_t*)(ptrs[KM]), (uint8_t*)(ptrs[VM]), (uint8_t*)(ptrs[RM]),
        (float*)(ptrs[KR]), (float*)(ptrs[VR]), (float*)(ptrs[RR]),
        (float*)(ptrs[O1]), (float*)(ptrs[O2]), (float*)(ptrs[O3]),
        (uint8_t*)(ptrs[ATTOUT]), (float*)(ptrs[ATTOUTR]), (float*)(ptrs[ATTOUTO]),
        (double*)(ptrs[FFNMIXK]), (double*)(ptrs[FFNMIXV]),
        (uint8_t*)(ptrs[FFNK]), (uint8_t*)(ptrs[FFNV]), (uint8_t*)(ptrs[FFNR]), 
        (float*)(ptrs[FFNKR]), (float*)(ptrs[FFNVR]), (float*)(ptrs[FFNRR]), 
        (float*)(ptrs[FFNKO]), (float*)(ptrs[FFNVO]), (float*)(ptrs[FFNRO]), 
        (double*)(ptrs[FFNKBUFFER]), (double*)(ptrs[FFNVBUFFER]), (float*)(ptrs[FFNRBUFFER]),
        (double*)(ptrs[DECAY]), (double*)(ptrs[BONUS]),
        (uint8_t*)(ptrs[HEAD]), (float*)(ptrs[HEADR]), (float*)(ptrs[HEADO])
        );
}
// function returns (int,int)

std::tuple<int64_t,int64_t> load (const std::string& filename) {
    std::ifstream binfile(filename, std::ios::in | std::ios::binary);
    if (!binfile.is_open()) {
        std::cout << "Error opening file " << filename << std::endl;
        exit(1);
    }

    // get n_layers
    // get n_embed
    int64_t n_layers, n_embed;
    binfile.read((char*)&n_layers, sizeof(int64_t));
    binfile.read((char*)&n_embed, sizeof(int64_t));
    // print
    std::cout << "n_layers: " << n_layers << std::endl;
    std::cout << "n_embed: " << n_embed << std::endl;
    num_embed = n_embed;
    num_layers = n_layers;

    for(int64_t i = 0; i < 46; i++) {
        int64_t size = getSize(i, n_layers, n_embed);
        if(types[i] == sizeof(double)){
            ptrs[i] = (int*)(new double[size]);
        } else if(types[i] == sizeof(float)) {
            ptrs[i] = (int*)(new float[size]);
        } else if(types[i] == sizeof(uint8_t)) {
            ptrs[i] = (int*)(new uint8_t[size]);
        } else {
            std::cout << "Error: size not supported" << std::endl;
            exit(1);
        }
        std::cout << "loading: " << names[i] << "\n";
        binfile.read((char*)(ptrs[i]), size*types[i]);
    }
    
    binfile.close();

    //   // return an array of pointers

    // return (n_layers, n_embed)
    return std::make_tuple(n_layers, n_embed);

}
void moveToCuda(int** ptrs, int64_t n_layers, int64_t n_embed);

void moveToCudaWrapper(){
    std::cout << "num_layers_cuda" << num_layers << std::endl;
    std::cout << "num_embed_cuda" << num_embed << std::endl;
    moveToCuda(ptrs, num_layers, num_embed);
}
void setState(int64_t n_embed, int64_t n_layers,
    double* stateaa, double* statebb, double* statecc, double* statedd, double* stateee,
    double* instateaa, double* instatebb, double* instatecc, double* instatedd, double* instateee);
// the only pytorch connection, comment out to use elsewhere
void attachState(torch::Tensor xy, torch::Tensor aa, torch::Tensor bb, torch::Tensor pp, torch::Tensor dd){
    enum {x,embed,layernorms,statexy,stateaa,statebb,statepp,statedd,buffer1,buffer2,buffer3,buffer4,mixk,mixv,mixr,km,vm,rm,kr,vr,rr,o1,o2,o3,attout,attoutr,attouto,ffnmixk,ffnmixv,ffnk,ffnv,ffnr,ffnkr,ffnvr,ffnrr,ffnko,ffnvo,ffnro,ffnkbuffer,ffnvbuffer,ffnrbuffer,decay,bonus,head,headr,heado};
    
    setState(num_embed, num_layers, 
        (double*)ptrs[statexy], (double*)ptrs[stateaa], (double*)ptrs[statebb], (double*)ptrs[statepp], (double*)ptrs[statedd],
        xy.data_ptr<double>(), aa.data_ptr<double>(), bb.data_ptr<double>(), pp.data_ptr<double>(), dd.data_ptr<double>()    
    );
    
}
void getOutput(float* in, float* out);
void getState(torch::Tensor xy, torch::Tensor aa, torch::Tensor bb, torch::Tensor pp, torch::Tensor dd, torch::Tensor out){
    enum {x,embed,layernorms,statexy,stateaa,statebb,statepp,statedd,buffer1,buffer2,buffer3,buffer4,mixk,mixv,mixr,km,vm,rm,kr,vr,rr,o1,o2,o3,attout,attoutr,attouto,ffnmixk,ffnmixv,ffnk,ffnv,ffnr,ffnkr,ffnvr,ffnrr,ffnko,ffnvo,ffnro,ffnkbuffer,ffnvbuffer,ffnrbuffer,decay,bonus,head,headr,heado};
    
    setState(num_embed, num_layers, 
    xy.data_ptr<double>(), aa.data_ptr<double>(), bb.data_ptr<double>(), pp.data_ptr<double>(), dd.data_ptr<double>(),
        (double*)ptrs[statexy], (double*)ptrs[stateaa], (double*)ptrs[statebb], (double*)ptrs[statepp], (double*)ptrs[statedd]
    );

    // copy buffer2 to out
    getOutput((float*)ptrs[buffer2],out.data_ptr<float>());
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rwkvc", &cuda_rwkv_wrapper, "rwkvc");
    m.def("load", &load, "load");
    m.def("toCuda", &moveToCudaWrapper, "toCuda");
    m.def("attachState", &attachState, "attachState");
    m.def("getState", &getState, "getState");

}

TORCH_LIBRARY(rwkv, m) {
    m.def("rwkvc", cuda_rwkv_wrapper);
    m.def("load", load);
    m.def("toCuda", moveToCudaWrapper);
    m.def("attachState", attachState);
    m.def("getState", getState);
}
