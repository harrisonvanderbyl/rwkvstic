#include <torch/extension.h>
#include "ATen/ATen.h"
#include <iostream>
#include <c10/cuda/CUDAGuard.h>
// generic T either float or fp16 or fp64
int** ptrs = new int*[46];
int num_layers = 0;
int num_embed = 0;
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
    enum {x,embed,layernorms,statexy,stateaa,statebb,statepp,statedd,buffer1,buffer2,buffer3,buffer4,mixk,mixv,mixr,km,vm,rm,kr,vr,rr,o1,o2,o3,attout,attoutr,attouto,ffnmixk,ffnmixv,ffnk,ffnv,ffnr,ffnkr,ffnvr,ffnrr,ffnko,ffnvo,ffnro,ffnkbuffer,ffnvbuffer,ffnrbuffer,decay,bonus,head,headr,heado};
    
    cuda_rwkv(num_layers, num_embed, token, (double*)(ptrs[x]),
        (float*)(ptrs[embed]), (double*)(ptrs[layernorms]),
        (double*)(ptrs[statexy]), (double*)(ptrs[stateaa]), (double*)(ptrs[statebb]), (double*)(ptrs[statepp]), (double*)(ptrs[statedd]),
        (double*)(ptrs[buffer1]), (float*)(ptrs[buffer2]), (float*)(ptrs[buffer3]), (float*)(ptrs[buffer4]),
        (double*)(ptrs[mixk]), (double*)(ptrs[mixv]), (double*)(ptrs[mixr]),
        (uint8_t*)(ptrs[km]), (uint8_t*)(ptrs[vm]), (uint8_t*)(ptrs[rm]),
        (float*)(ptrs[kr]), (float*)(ptrs[vr]), (float*)(ptrs[rr]),
        (float*)(ptrs[o1]), (float*)(ptrs[o2]), (float*)(ptrs[o3]),
        (uint8_t*)(ptrs[attout]), (float*)(ptrs[attoutr]), (float*)(ptrs[attouto]),
        (double*)(ptrs[ffnmixk]), (double*)(ptrs[ffnmixv]),
        (uint8_t*)(ptrs[ffnk]), (uint8_t*)(ptrs[ffnv]), (uint8_t*)(ptrs[ffnr]), 
        (float*)(ptrs[ffnkr]), (float*)(ptrs[ffnvr]), (float*)(ptrs[ffnrr]), 
        (float*)(ptrs[ffnko]), (float*)(ptrs[ffnvo]), (float*)(ptrs[ffnro]), 
        (double*)(ptrs[ffnkbuffer]), (double*)(ptrs[ffnvbuffer]), (float*)(ptrs[ffnrbuffer]),
        (double*)(ptrs[decay]), (double*)(ptrs[bonus]),
        (uint8_t*)(ptrs[head]), (float*)(ptrs[headr]), (float*)(ptrs[heado])
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

    // get xbuf
    double* xbuf = new double[n_embed];
    binfile.read((char*)xbuf, n_embed*sizeof(double));
    // print
    std::cout << "xbuf: " << xbuf[0] << ":" << xbuf[n_embed-1] << std::endl;


    // get embed 
    float* embed = new float[n_embed*50277];
    binfile.read((char*)embed, n_embed*50277*sizeof(float));
    // print
    std::cout << "embed: " << embed[0] << ":" << embed[50277*n_embed-1] << std::endl;

    // get layernorms
    double* layernorms = new double[(n_layers+1)*4*n_embed];
    binfile.read((char*)layernorms, (n_layers+1)*4*n_embed*sizeof(double));
    // print
    std::cout << "layernorms: " << layernorms[0] << ":" << layernorms[(n_layers+1)*n_embed*4-1] << std::endl;

    // get statexy
    double* statexy = new double[n_layers*n_embed];
    binfile.read((char*)statexy, n_layers*n_embed*sizeof(double));
    // print
    std::cout << "statexy: " << statexy[0] << ":" << statexy[n_layers*n_embed-1] << std::endl;

    // get stateaa
    double* stateaa = new double[n_layers*n_embed];
    binfile.read((char*)stateaa, n_layers*n_embed*sizeof(double));
    // print
    std::cout << "stateaa: " << stateaa[0] << ":" << stateaa[n_layers*n_embed-1] << std::endl;

    // get statebb
    double* statebb = new double[n_layers*n_embed];
    binfile.read((char*)statebb, n_layers*n_embed*sizeof(double));
    // print
    std::cout << "statebb: " << statebb[0] << ":" << statebb[n_layers*n_embed-1] << std::endl;

    // get statepp
    double* statepp = new double[n_layers*n_embed];
    binfile.read((char*)statepp, n_layers*n_embed*sizeof(double));
    // print
    std::cout << "statepp: " << statepp[0] << ":" << statepp[n_layers*n_embed-1] << std::endl;

    // get statedd
    double* statedd = new double[n_layers*n_embed];
    binfile.read((char*)statedd, n_layers*n_embed*sizeof(double));
    // print
    std::cout << "statedd: " << statedd[0] << ":" << statedd[n_layers*n_embed-1] << std::endl;

    // get buffer 1
    double* buffer1 = new double[n_embed];
    binfile.read((char*)buffer1, n_embed*sizeof(double));
    // print
    std::cout << "buffer1: " << buffer1[0] << ":" << buffer1[n_embed-1] << std::endl;

    // get buffer 2
    float* buffer2 = new float[50277];
    binfile.read((char*)buffer2, 50277*sizeof(float));
    // print
    std::cout << "buffer2: " << buffer2[0] << ":" << buffer2[50277-1] << std::endl;

    // get buffer 3
    float* buffer3 = new float[n_embed];
    binfile.read((char*)buffer3, n_embed*sizeof(float));
    // print
    std::cout << "buffer3: " << buffer3[0] << ":" << buffer3[n_embed-1] << std::endl;

    // get buffer 4
    float* buffer4 = new float[n_embed];
    binfile.read((char*)buffer4, n_embed*sizeof(float));
    // print
    std::cout << "buffer4: " << buffer4[0] << ":" << buffer4[n_embed-1] << std::endl;

    // get mixk, mixv, mixr
    double* mixk = new double[n_layers*n_embed];
    binfile.read((char*)mixk, n_layers*n_embed*sizeof(double));
    // print
    std::cout << "mixk: " << mixk[0] << ":" << mixk[n_layers*n_embed-1] << std::endl;

    double* mixv = new double[n_layers*n_embed];
    binfile.read((char*)mixv, n_layers*n_embed*sizeof(double));
    // print
    std::cout << "mixv: " << mixv[0] << ":" << mixv[n_layers*n_embed-1] << std::endl;

    double* mixr = new double[n_layers*n_embed];
    binfile.read((char*)mixr, n_layers*n_embed*sizeof(double));
    // print
    std::cout << "mixr: " << mixr[0] << ":" << mixr[n_layers*n_embed-1] << std::endl;

    // get km, vm, rm uint8 matruxs of n_layers*n_embed*n_embed
    uint8_t* km = new uint8_t[n_layers*n_embed*n_embed];
    binfile.read((char*)km, n_layers*n_embed*n_embed*sizeof(uint8_t));
    // print
    std::cout << "km: " << (int)km[0] << ":" << (int)km[n_layers*n_embed*n_embed-1] << std::endl;

    uint8_t* vm = new uint8_t[n_layers*n_embed*n_embed];
    binfile.read((char*)vm, n_layers*n_embed*n_embed*sizeof(uint8_t));
    // print
    std::cout << "vm: " << (int)vm[0] << ":" << (int)vm[n_layers*n_embed*n_embed-1] << std::endl;

    uint8_t* rm = new uint8_t[n_layers*n_embed*n_embed];
    binfile.read((char*)rm, n_layers*n_embed*n_embed*sizeof(uint8_t));
    // print
    std::cout << "rm: " << (int)rm[0] << ":" << (int)rm[n_layers*n_embed*n_embed-1] << std::endl;

    // get ranges for km, vm, rm of n_layers*n_embed float
    float* krange = new float[n_layers*n_embed];
    binfile.read((char*)krange, n_layers*n_embed*sizeof(float));
    // print
    std::cout << "krange: " << krange[0] << ":" << krange[n_layers*n_embed-1] << std::endl;

    float* vrange = new float[n_layers*n_embed];
    binfile.read((char*)vrange, n_layers*n_embed*sizeof(float));
    // print
    std::cout << "vrange: " << vrange[0] << ":" << vrange[n_layers*n_embed-1] << std::endl;

    float* rrange = new float[n_layers*n_embed];
    binfile.read((char*)rrange, n_layers*n_embed*sizeof(float));
    // print
    std::cout << "rrange: " << rrange[0] << ":" << rrange[n_layers*n_embed-1] << std::endl;

    // get bias for km, vm, rm of n_layers*n_embed float
    float* kbias = new float[n_layers*n_embed];
    binfile.read((char*)kbias, n_layers*n_embed*sizeof(float));
    // print
    std::cout << "kbias: " << kbias[0] << ":" << kbias[n_layers*n_embed-1] << std::endl;

    float* vbias = new float[n_layers*n_embed];
    binfile.read((char*)vbias, n_layers*n_embed*sizeof(float));
    // print
    std::cout << "vbias: " << vbias[0] << ":" << vbias[n_layers*n_embed-1] << std::endl;

    float* rbias = new float[n_layers*n_embed];
    binfile.read((char*)rbias, n_layers*n_embed*sizeof(float));
    // print
    std::cout << "rbias: " << rbias[0] << ":" << rbias[n_layers*n_embed-1] << std::endl;

    // remaining
    // attout uint8_t n_layers*n_embed*n_embed
    // attoutr float n_layers*n_embed
    // attouto float n_layers*n_embed
    // ffnmixk double n_layers*n_embed
    // ffnmixv double n_layers*n_embed
    // ffnk uint8_t n_layers*n_embed*n_embed
    // ffnv uint8_t n_layers*n_embed*n_embed*4
    // ffnr uint8_t n_layers*n_embed*n_embed*4
    

    uint8_t* attout = new uint8_t[n_layers*n_embed*n_embed];
    binfile.read((char*)attout, n_layers*n_embed*n_embed*sizeof(uint8_t));
    // print
    std::cout << "attout: " << (int)attout[0] << ":" << (int)attout[n_layers*n_embed*n_embed-1] << std::endl;

    float* attoutr = new float[n_layers*n_embed];
    binfile.read((char*)attoutr, n_layers*n_embed*sizeof(float));
    // print
    std::cout << "attoutr: " << attoutr[0] << ":" << attoutr[n_layers*n_embed-1] << std::endl;

    float* attouto = new float[n_layers*n_embed];
    binfile.read((char*)attouto, n_layers*n_embed*sizeof(float));
    // print
    std::cout << "attouto: " << attouto[0] << ":" << attouto[n_layers*n_embed-1] << std::endl;

    double* ffnmixk = new double[n_layers*n_embed];
    binfile.read((char*)ffnmixk, n_layers*n_embed*sizeof(double));
    // print
    std::cout << "ffnmixk: " << ffnmixk[0] << ":" << ffnmixk[n_layers*n_embed-1] << std::endl;

    double* ffnmixv = new double[n_layers*n_embed];
    binfile.read((char*)ffnmixv, n_layers*n_embed*sizeof(double));
    // print
    std::cout << "ffnmixv: " << ffnmixv[0] << ":" << ffnmixv[n_layers*n_embed-1] << std::endl;

    uint8_t* ffnk = new uint8_t[n_layers*n_embed*n_embed];
    binfile.read((char*)ffnk, n_layers*n_embed*n_embed*sizeof(uint8_t));
    // print
    std::cout << "ffnk: " << (int)ffnk[0] << ":" << (int)ffnk[n_layers*n_embed*n_embed-1] << std::endl;

    uint8_t* ffnv = new uint8_t[n_layers*n_embed*n_embed*4];
    binfile.read((char*)ffnv, n_layers*n_embed*n_embed*4*sizeof(uint8_t));
    // print
    std::cout << "ffnv: " << (int)ffnv[0] << ":" << (int)ffnv[n_layers*n_embed*n_embed*4-1] << std::endl;

    uint8_t* ffnr = new uint8_t[n_layers*n_embed*n_embed*4];
    binfile.read((char*)ffnr, n_layers*n_embed*n_embed*4*sizeof(uint8_t));
    // print
    std::cout << "ffnr: " << (int)ffnr[0] << ":" << (int)ffnr[n_layers*n_embed*n_embed*4-1] << std::endl;

    // ffnkr float n_layers*n_embed
    // ffnvr float n_layers*n_embed*4
    // ffnrr float n_layers*n_embed
    // ffnko float n_layers*n_embed
    // ffnvo float n_layers*n_embed*4
    // ffnro float n_layers*n_embed
    // ffnkbuffer double n_embed
    // ffnvbuffer double n_embed
    // decay double n_layers*n_embed
    // bonus double n_layers*n_embed
    // head uint8_t 50277*n_embed*n_embed
    // headr float 50277*n_embed
    // heado float 50277*n_embed

    float* ffnkr = new float[n_layers*n_embed];
    binfile.read((char*)ffnkr, n_layers*n_embed*sizeof(float));
    // print
    std::cout << "ffnkr: " << ffnkr[0] << ":" << ffnkr[n_layers*n_embed-1] << std::endl;

    float* ffnvr = new float[n_layers*n_embed*4];
    binfile.read((char*)ffnvr, n_layers*n_embed*4*sizeof(float));
    // print
    std::cout << "ffnvr: " << ffnvr[0] << ":" << ffnvr[n_layers*n_embed*4-1] << std::endl;

    float* ffnrr = new float[n_layers*n_embed];
    binfile.read((char*)ffnrr, n_layers*n_embed*sizeof(float));
    // print
    std::cout << "ffnrr: " << ffnrr[0] << ":" << ffnrr[n_layers*n_embed-1] << std::endl;

    float* ffnko = new float[n_layers*n_embed];
    binfile.read((char*)ffnko, n_layers*n_embed*sizeof(float));
    // print
    std::cout << "ffnko: " << ffnko[0] << ":" << ffnko[n_layers*n_embed-1] << std::endl;

    float* ffnvo = new float[n_layers*n_embed*4];
    binfile.read((char*)ffnvo, n_layers*n_embed*4*sizeof(float));
    // print
    std::cout << "ffnvo: " << ffnvo[0] << ":" << ffnvo[n_layers*n_embed*4-1] << std::endl;

    float* ffnro = new float[n_layers*n_embed];
    binfile.read((char*)ffnro, n_layers*n_embed*sizeof(float));
    // print
    std::cout << "ffnro: " << ffnro[0] << ":" << ffnro[n_layers*n_embed-1] << std::endl;

    double* ffnkbuffer = new double[n_embed];
    binfile.read((char*)ffnkbuffer, n_embed*sizeof(double));
    // print
    std::cout << "ffnkbuffer: " << ffnkbuffer[0] << ":" << ffnkbuffer[n_embed-1] << std::endl;
    
    double* ffnvbuffer = new double[n_embed];
    binfile.read((char*)ffnvbuffer, n_embed*sizeof(double));
    // print
    std::cout << "ffnvbuffer: " << ffnvbuffer[0] << ":" << ffnvbuffer[n_embed-1] << std::endl;

    float* ffnrbuffer = new float[n_embed*4];
    binfile.read((char*)ffnrbuffer, n_embed*4*sizeof(float));
    // print
    std::cout << "ffnrbuffer: " << ffnrbuffer[0] << ":" << ffnrbuffer[n_embed*4-1] << std::endl;

    double* decay = new double[n_layers*n_embed];
    binfile.read((char*)decay, n_layers*n_embed*sizeof(double));
    // print
    std::cout << "decay: " << decay[0] << ":" << decay[n_layers*n_embed-1] << std::endl;

    double* bonus = new double[n_layers*n_embed];
    binfile.read((char*)bonus, n_layers*n_embed*sizeof(double));
    // print
    std::cout << "bonus: " << bonus[0] << ":" << bonus[n_layers*n_embed-1] << std::endl;

    uint8_t* head = new uint8_t[50277*n_embed];
    binfile.read((char*)head, 50277*n_embed*sizeof(uint8_t));
    // print
    std::cout << "head: " << (int)head[0] << ":" << (int)head[50277*n_embed-1] << std::endl;

    float* headr = new float[n_embed];
    binfile.read((char*)headr, n_embed*sizeof(float));
    // print
    std::cout << "headr: " << headr[0] << ":" << headr[n_embed-1] << std::endl;

    float* heado = new float[n_embed];
    binfile.read((char*)heado, n_embed*sizeof(float));
    // print
    std::cout << "heado: " << heado[0] << ":" << heado[n_embed-1] << std::endl;


    binfile.close();

    //   // return an array of pointers
    
    ptrs[0] = (int*)xbuf;
    ptrs[1] = (int*)embed;
    ptrs[2] = (int*)layernorms;
    ptrs[3] = (int*)statexy;
    ptrs[4] = (int*)stateaa;
    ptrs[5] = (int*)statebb;
    ptrs[6] = (int*)statepp;
    ptrs[7] = (int*)statedd;
    ptrs[8] = (int*)buffer1;
    ptrs[9] = (int*)buffer2;
    ptrs[10] = (int*)buffer3;
    ptrs[11] = (int*)buffer4;
    ptrs[12] = (int*)mixk;
    ptrs[13] = (int*)mixv;
    ptrs[14] = (int*)mixr;
    ptrs[15] = (int*)km;
    ptrs[16] = (int*)vm;  
    ptrs[17] = (int*)rm;
    ptrs[18] = (int*)krange;
    ptrs[19] = (int*)vrange;
    ptrs[20] = (int*)rrange;
    ptrs[21] = (int*)kbias;
    ptrs[22] = (int*)vbias;
    ptrs[23] = (int*)rbias;
    ptrs[24] = (int*)attout;  
    ptrs[25] = (int*)attoutr;
    ptrs[26] = (int*)attouto;
    ptrs[27] = (int*)ffnmixk;
    ptrs[28] = (int*)ffnmixv;
    ptrs[29] = (int*)ffnk;
    ptrs[30] = (int*)ffnv;
    ptrs[31] = (int*)ffnr;
    ptrs[32] = (int*)ffnkr;
    ptrs[33] = (int*)ffnvr;
    ptrs[34] = (int*)ffnrr;
    ptrs[35] = (int*)ffnko;
    ptrs[36] = (int*)ffnvo;
    ptrs[37] = (int*)ffnro;
    ptrs[38] = (int*)ffnkbuffer;
    ptrs[39] = (int*)ffnvbuffer;
    ptrs[40] = (int*)ffnrbuffer;
    ptrs[41] = (int*)decay;
    ptrs[42] = (int*)bonus;
    ptrs[43] = (int*)head;
    ptrs[44] = (int*)headr;
    ptrs[45] = (int*)heado;

    num_embed = n_embed;
    num_layers = n_layers;

    // return (n_layers, n_embed)
    return std::make_tuple(n_layers, n_embed);

}
void moveToCuda(int** ptrs, int64_t n_layers, int64_t n_embed);

void moveToCudaWrapper(){
    moveToCuda(ptrs, num_layers, num_embed);
}

// the only pytorch connection, comment out to use elsewhere
void attachState(torch::Tensor xy, torch::Tensor aa, torch::Tensor bb, torch::Tensor pp, torch::Tensor dd, torch::Tensor out){
    enum {x,embed,layernorms,statexy,stateaa,statebb,statepp,statedd,buffer1,buffer2,buffer3,buffer4,mixk,mixv,mixr,km,vm,rm,kr,vr,rr,o1,o2,o3,attout,attoutr,attouto,ffnmixk,ffnmixv,ffnk,ffnv,ffnr,ffnkr,ffnvr,ffnrr,ffnko,ffnvo,ffnro,ffnkbuffer,ffnvbuffer,ffnrbuffer,decay,bonus,head,headr,heado};
    
    ptrs[statexy] = (int*)xy.data_ptr();
    ptrs[stateaa] = (int*)aa.data_ptr();
    ptrs[statebb] = (int*)bb.data_ptr();
    ptrs[statepp] = (int*)pp.data_ptr();
    ptrs[statedd] = (int*)dd.data_ptr();
    ptrs[buffer2] = (int*)out.data_ptr();
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rwkvc", &cuda_rwkv_wrapper, "rwkvc");
    m.def("load", &load, "load");
    m.def("toCuda", &moveToCudaWrapper, "toCuda");
    m.def("attachState", &attachState, "attachState");

}

TORCH_LIBRARY(rwkv, m) {
    m.def("rwkvc", cuda_rwkv_wrapper);
    m.def("load", load);
    m.def("toCuda", moveToCudaWrapper);
    m.def("attachState", attachState);
}
