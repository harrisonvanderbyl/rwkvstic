// include std
#include <cstdint>
enum {X,EMBED,LAYERNORMS,STATEXY,STATEAA,STATEBB,STATEPP,STATEDD,BUFFER1,BUFFER2,BUFFER3,BUFFER4,MIXK,MIXV,MIXR,KM,VM,RM,KR,VR,RR,O1,O2,O3,ATTOUT,ATTOUTR,ATTOUTO,FFNMIXK,FFNMIXV,FFNK,FFNV,FFNR,FFNKR,FFNVR,FFNRR,FFNKO,FFNVO,FFNRO,FFNKBUFFER,FFNVBUFFER,FFNRBUFFER,DECAY,BONUS,HEAD,HEADR,HEADO};

int64_t getSize(int64_t i, int64_t a, int64_t b) {
  int64_t sizes[46] = {b,50277*b,4*(a+1)*b, a*b,a*b,a*b,a*b,a*b, b, 50277,b,b,a*b,a*b,a*b,a*b*b,a*b*b,a*b*b,a*b,a*b,a*b,a*b,a*b,a*b,a*b*b,a*b,a*b,a*b,a*b,a*b*b*4,a*b*b*4,a*b*b,a*b,a*b*4,a*b,a*b,a*b*4,a*b,b,b,b*4,a*b,a*b,50277*b,b,b};
    return sizes[i];
}


unsigned long f = sizeof(float);
unsigned long d = sizeof(double);
unsigned long g = sizeof(uint8_t);

unsigned long types[46] = {d,f,d,d,d,d,d,d,d,f,f,f,d,d,d,g,g,g,f,f,f,f,f,f,g,f,f,d,d,g,g,g,f,f,f,f,f,f,d,d,f,d,d,g,f,f};

unsigned long Mtypes(int64_t i) {
    return types[i];
}

char* names[46] = {
    "xbuf",
    "embed",
    "layernorms",
    "state_xy",
    "state_aa",
    "state_bb",
    "state_pp",
    "state_dd",
    "buffer1",
    "buffer2",
    "buffer3",
    "buffer4",
    "mix_k",
    "mix_v",
    "mix_r",
    "km",
    "vm",
    "rm",
    "kr",
    "vr",
    "rr",
    "o1",
    "o2",
    "o3",
    "att_out",
    "att_out_r",
    "att_out_o",
    "ffn_mix_k",
    "ffn_mix_v",
    "ffn_k",
    "ffn_v",
    "ffn_r",
    "ffn_kr",
    "ffn_vr",
    "ffn_rr",
    "ffn_ko",
    "ffn_vo",
    "ffn_ro",
    "ffn_k_buffer",
    "ffn_v_buffer",
    "ffn_r_buffer",
    "decay",
    "bonus",
    "head",
    "head_r",
    "head_o"

};