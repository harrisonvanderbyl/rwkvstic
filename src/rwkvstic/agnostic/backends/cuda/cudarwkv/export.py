import torch
from tqdm import tqdm

import os

current_path = os.path.dirname(os.path.abspath(__file__))


def OptRWKV(path):
    
    from torch.utils.cpp_extension import load
    import os
    current_path = os.path.dirname(os.path.abspath(__file__))
    load(
        name=f"wkv_cuda",
        sources=[f"{current_path}/export.cpp"
                ],
        )
    

    class myRWKV(torch.nn.Module):
        def QuantizeMatrix(self, xx):
                x = xx
                rang = 255
                mini = x.min(0)[0].double()
                out = x-mini
                ran = out.max(0)[0].double()/rang
                out = out/ran
                fractmat = out.frac().double()
                fractmat = fractmat.mean(0)
                mini = mini.double() + fractmat*ran.double()
                
                return [out.t(),ran.to(torch.float32).clone(), mini.to(torch.float32).clone()]
        def __init__(self,w,dims,layers):
            super(myRWKV, self).__init__()
            
            self.emptyState = layers * [[[0]*dims]*4+[[-1e30]*dims]]
            vustatea = torch.tensor([self.emptyState[i][0] for i in range(layers)]).double().contiguous()
            vustateb = torch.tensor([self.emptyState[i][2] for i in range(layers)]).double().contiguous()
            vustatec = torch.tensor([self.emptyState[i][3] for i in range(layers)]).double().contiguous()
            vustated = torch.tensor([self.emptyState[i][4] for i in range(layers)]).double().contiguous()
            vustatee = torch.tensor([self.emptyState[i][1] for i in range(layers)]).double().contiguous()
            self.emptyState = [vustatea,vustateb,vustatec,vustated,vustatee]

            self.emb =  w["emb.weight"].float().contiguous()
            
            
            sn = ["blocks.0.ln0.weight","blocks.0.ln0.bias"]
            for i in range(layers):
                sn.append(f"blocks.{i}.ln1.weight")
                sn.append(f"blocks.{i}.ln1.bias")
                sn.append(f"blocks.{i}.ln2.weight")
                sn.append(f"blocks.{i}.ln2.bias")
            sn += [
                "ln_out.weight",
                "ln_out.bias"
            ]
            
            self.cudalnin = torch.stack (
                [w[x] for x in sn]).double().contiguous()
            self.mixk = [w[f"blocks.{i}.att.time_mix_k"].squeeze() for i in range(layers)]
            self.mixk = torch.stack(self.mixk).double().contiguous()
            self.mixv = [w[f"blocks.{i}.att.time_mix_v"].squeeze() for i in range(layers)]
            self.mixv = torch.stack(self.mixv).double().contiguous()
            self.mixr = [w[f"blocks.{i}.att.time_mix_r"].squeeze() for i in range(layers)]
            self.mixr = torch.stack(self.mixr).double().contiguous()
            self.mixffnk = [w[f"blocks.{i}.ffn.time_mix_k"].squeeze() for i in range(layers)]
            self.mixffnk = torch.stack(self.mixffnk).double().contiguous()
            self.mixffnr = [w[f"blocks.{i}.ffn.time_mix_r"].squeeze() for i in range(layers)]
            self.mixffnr = torch.stack(self.mixffnr).double().contiguous()
            self.decay = [w[f"blocks.{i}.att.time_decay"].squeeze() for i in range(layers)]
            self.decay = torch.stack(self.decay).double().contiguous().exp().neg()
            self.bonus = [w[f"blocks.{i}.att.time_first"].squeeze() for i in range(layers)]
            self.bonus = torch.stack(self.bonus).double().contiguous()

            toQuantize = [
                "att.key.weight",
                "att.value.weight",
                "att.receptance.weight",
                "ffn.key.weight",
                "ffn.value.weight",
                "ffn.receptance.weight",
                "att.output.weight"
            ]
            for key in toQuantize:
                weights =[self.QuantizeMatrix(w[f"blocks.{i}.{key}"] ) for i in tqdm(range(layers), desc=f"Quantizing {key}")]
                print("stacking weights")
                keysp = key.split(".")[:2]
                keysp = "".join(keysp)                
                self.__dict__[keysp+"weights"] = torch.stack([x[0] for x in weights]).to(dtype=torch.uint8, memory_format=torch.contiguous_format)
                self.__dict__[keysp+"ranges"] = torch.stack([x[1] for x in weights]).to(dtype=torch.float32, memory_format=torch.contiguous_format)
                self.__dict__[keysp+"zp"] = torch.stack([x[2] for x in weights]).to(dtype=torch.float32, memory_format=torch.contiguous_format)
                import gc
                for x in tqdm(range(layers), desc=f"Cleaning {key}"):
                    del w[f"blocks.{x}.{key}"]
                del weights 
                gc.collect()
                torch.cuda.empty_cache()
            
            
            self.cudahead, self.cudaheadr, self.cudaheadzp = self.QuantizeMatrix(w["head.weight"])
            self.cudahead = self.cudahead.to(dtype=torch.uint8, memory_format=torch.contiguous_format)
            del w

            self.dim = dims
            self.layers = layers
            
            self.rx = torch.zeros((self.dim), dtype = torch.double)
            self.buffer0 = torch.zeros(self.dim, dtype=torch.double)
            self.buffer1 = torch.zeros(50277, dtype=torch.float)
            self.buffer2 = torch.zeros(self.dim, dtype=torch.float)
            self.buffer3 = torch.zeros(self.dim, dtype=torch.float)
            self.ffnvbuf = torch.zeros(self.dim, dtype=torch.double)
            self.ffnkbuf = torch.zeros(self.dim, dtype=torch.double)
            self.ffkeybuffer = torch.zeros(self.dim*4, dtype=torch.float)       

            
        def forward(self):
            
            torch.ops.rwkv.save(
                self.layers,
                self.dim,
                self.rx,
                self.emb,
                self.cudalnin,
                self.emptyState[0],
                self.emptyState[1],
                self.emptyState[2],
                self.emptyState[3],
                self.emptyState[4],
                self.buffer0,
                self.buffer1,
                self.buffer2,
                self.buffer3,
                self.mixk,
                self.mixv,
                self.mixr,
                self.attkeyweights,
                self.attvalueweights,
                self.attreceptanceweights,
                self.attkeyranges,
                self.attvalueranges,
                self.attreceptanceranges,
                self.attkeyzp,
                self.attvaluezp,
                self.attreceptancezp,
                self.attoutputweights,
                self.attoutputranges,
                self.attoutputzp,
                self.mixffnk,
                self.mixffnr,
                self.ffnkeyweights,
                self.ffnvalueweights,
                self.ffnreceptanceweights,
                self.ffnkeyranges,
                self.ffnvalueranges,
                self.ffnreceptanceranges,
                self.ffnkeyzp,
                self.ffnvaluezp,
                self.ffnreceptancezp,
                self.ffnkbuf,
                self.ffnvbuf,
                self.ffkeybuffer,
                self.decay,
                self.bonus,
                self.cudahead,
                self.cudaheadr,
                self.cudaheadzp
            )
            print("embed: ",self.emb[0][0],":", self.emb[-1][-1])
            print("layernorms:", self.cudalnin[0][0],":", self.cudalnin[-1][-1])
            print("emptystate0:", self.emptyState[0][0][0],":", self.emptyState[0][0][-1])
            print("emptystate1:", self.emptyState[1][0][0],":", self.emptyState[1][0][-1])
            print("emptystate2:", self.emptyState[2][0][0],":", self.emptyState[2][0][-1])
            print("emptystate3:", self.emptyState[3][0][0],":", self.emptyState[3][0][-1])
            print("emptystate4:", self.emptyState[4][0][0],":", self.emptyState[4][0][-1])
            print("mixk:", self.mixk[0][0],":", self.mixk[-1][-1])
            print("mixv:", self.mixv[0][0],":", self.mixv[-1][-1])
            print("mixr:", self.mixr[0][0],":", self.mixr[-1][-1])
            print("attkeyweights:", self.attkeyweights[0][0][0],":", self.attkeyweights[-1][-1][-1])
            print("attvalueweights:", self.attvalueweights[0][0][0],":", self.attvalueweights[-1][-1][-1])
            print("attreceptanceweights:", self.attreceptanceweights[0][0][0],":", self.attreceptanceweights[-1][-1][-1])
            print("attkeyranges:", self.attkeyranges[0][0],":", self.attkeyranges[-1][-1])
            print("attvalueranges:", self.attvalueranges[0][0],":", self.attvalueranges[-1][-1])
            print("attreceptanceranges:", self.attreceptanceranges[0][0],":", self.attreceptanceranges[-1][-1])
            print("attkeyzp:", self.attkeyzp[0][0],":", self.attkeyzp[-1][-1])
            print("attvaluezp:", self.attvaluezp[0][0],":", self.attvaluezp[-1][-1])
            print("attreceptancezp:", self.attreceptancezp[0][0],":", self.attreceptancezp[-1][-1])
            print("attoutputweights:", self.attoutputweights[0][0][0],":", self.attoutputweights[-1][-1][-1])
            print("attoutputranges:", self.attoutputranges[0][0],":", self.attoutputranges[-1][-1])
            print("attoutputzp:", self.attoutputzp[0][0],":", self.attoutputzp[-1][-1])
            print("mixffnk:", self.mixffnk[0][0],":", self.mixffnk[-1][-1])
            print("mixffnr:", self.mixffnr[0][0],":", self.mixffnr[-1][-1])
            print("ffnkeyweights:", self.ffnkeyweights[0][0][0],":", self.ffnkeyweights[-1][-1][-1])
            print("ffnvalueweights:", self.ffnvalueweights[0][0][0],":", self.ffnvalueweights[-1][-1][-1])
            print("ffnreceptanceweights:", self.ffnreceptanceweights[0][0][0],":", self.ffnreceptanceweights[-1][-1][-1])
            print("ffnkeyranges:", self.ffnkeyranges[0][0],":", self.ffnkeyranges[-1][-1])
            print("ffnvalueranges:", self.ffnvalueranges[0][0],":", self.ffnvalueranges[-1][-1])
            print("ffnreceptanceranges:", self.ffnreceptanceranges[0][0],":", self.ffnreceptanceranges[-1][-1])
            print("ffnkeyzp:", self.ffnkeyzp[0][0],":", self.ffnkeyzp[-1][-1])
            print("ffnvaluezp:", self.ffnvaluezp[0][0],":", self.ffnvaluezp[-1][-1])
            print("ffnreceptancezp:", self.ffnreceptancezp[0][0],":", self.ffnreceptancezp[-1][-1])
            print("ffnkbuf:", self.ffnkbuf[0],":", self.ffnkbuf[-1])
            print("ffnvbuf:", self.ffnvbuf[0],":", self.ffnvbuf[-1])
            print("ffkeybuffer:", self.ffkeybuffer[0],":", self.ffkeybuffer[-1])
            print("decay:", self.decay[0][0],":", self.decay[-1][-1])
            print("bonus:", self.bonus[0][0],":", self.bonus[-1][-1])
            print("cudahead:", self.cudahead[0][0],":", self.cudahead[-1][-1])
            print("cudaheadr:", self.cudaheadr[0],":", self.cudaheadr[-1])
            print("cudaheadzp:", self.cudaheadzp[0],":", self.cudaheadzp[-1])
            

            
            
        
       
            
        
        
    
    w = torch.load(path, map_location="cpu")
    # detach weights

    dims = len(w["blocks.0.att.key.weight"])
    layers = len(
        list(filter(lambda x: "blocks" in x and "ln1.bias" in x, w.keys())))

    myrwkv = myRWKV(w,dims,layers)
    myrwkv.forward()
    exit()
