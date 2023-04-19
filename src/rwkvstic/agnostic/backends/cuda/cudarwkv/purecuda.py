import torch
from tqdm import tqdm

import os

current_path = os.path.dirname(os.path.abspath(__file__))


def OptRWKV(path, jit=True, export=False,**kwargs):
    

    from rwkvstic.agnostic.backends.cuda.cudarwkv.load import loadModule
    loadModule()

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
                
                return [out.t(),ran.to(torch.float32).cuda().clone(), mini.to(torch.float32).cuda().clone()]
        def __init__(self,w,dims,layers):
            super(myRWKV, self).__init__()
            
            self.emptyState = layers * [[[0]*dims]*4+[[-1e30]*dims]]
            vustatea = torch.tensor([self.emptyState[i][0] for i in range(layers)]).double().contiguous().cuda()
            vustateb = torch.tensor([self.emptyState[i][2] for i in range(layers)]).double().contiguous().cuda()
            vustatec = torch.tensor([self.emptyState[i][3] for i in range(layers)]).double().contiguous().cuda()
            vustated = torch.tensor([self.emptyState[i][4] for i in range(layers)]).double().contiguous().cuda()
            vustatee = torch.tensor([self.emptyState[i][1] for i in range(layers)]).double().contiguous().cuda()
            self.emptyState = [vustatea,vustateb,vustatec,vustated,vustatee]

            self.emb =  w["emb.weight"].clone()
            
            
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
                [w[x] for x in sn]).double().contiguous().cuda()
            self.mixk = [w[f"blocks.{i}.att.time_mix_k"].squeeze() for i in range(layers)]
            self.mixk = torch.stack(self.mixk).double().contiguous().cuda()
            self.mixv = [w[f"blocks.{i}.att.time_mix_v"].squeeze() for i in range(layers)]
            self.mixv = torch.stack(self.mixv).double().contiguous().cuda()
            self.mixr = [w[f"blocks.{i}.att.time_mix_r"].squeeze() for i in range(layers)]
            self.mixr = torch.stack(self.mixr).double().contiguous().cuda()
            self.mixffnk = [w[f"blocks.{i}.ffn.time_mix_k"].squeeze() for i in range(layers)]
            self.mixffnk = torch.stack(self.mixffnk).double().contiguous().cuda()
            self.mixffnr = [w[f"blocks.{i}.ffn.time_mix_r"].squeeze() for i in range(layers)]
            self.mixffnr = torch.stack(self.mixffnr).double().contiguous().cuda()
            self.decay = [w[f"blocks.{i}.att.time_decay"].squeeze() for i in range(layers)]
            self.decay = torch.stack(self.decay).double().contiguous().exp().neg().cuda()
            self.bonus = [w[f"blocks.{i}.att.time_first"].squeeze() for i in range(layers)]
            self.bonus = torch.stack(self.bonus).double().contiguous().cuda()

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
                self.__dict__[keysp+"weights"] = torch.stack([x[0] for x in weights]).to(dtype=torch.uint8, memory_format=torch.contiguous_format, device="cuda")
                self.__dict__[keysp+"ranges"] = torch.stack([x[1] for x in weights]).to(dtype=torch.float32, memory_format=torch.contiguous_format, device="cuda")
                self.__dict__[keysp+"zp"] = torch.stack([x[2] for x in weights]).to(dtype=torch.float32, memory_format=torch.contiguous_format, device="cuda")
                import gc
                for x in tqdm(range(layers), desc=f"Cleaning {key}"):
                    del w[f"blocks.{x}.{key}"]
                del weights 
                gc.collect()
                torch.cuda.empty_cache()
            
            
            self.cudahead, self.cudaheadr, self.cudaheadzp = self.QuantizeMatrix(w["head.weight"])
            self.cudahead = self.cudahead.to(dtype=torch.uint8, memory_format=torch.contiguous_format, device="cuda")
            del w

            self.dim = dims
            self.layers = layers
            
            self.rx = torch.zeros((self.dim), dtype = torch.double, device = "cuda")
            self.buffer0 = torch.zeros(self.dim, dtype=torch.double, device= "cuda")
            self.buffer1 = torch.zeros(50277, dtype=torch.float, device= "cuda")
            self.buffer2 = torch.zeros(self.dim, dtype=torch.float, device= "cuda")
            self.buffer3 = torch.zeros(self.dim, dtype=torch.float, device= "cuda")
            self.ffnvbuf = torch.zeros(self.dim, dtype=torch.double, device= "cuda")
            self.ffnkbuf = torch.zeros(self.dim, dtype=torch.double, device= "cuda")
            self.ffkeybuffer = torch.zeros(self.dim*4, dtype=torch.float, device= "cuda")       

            
        def forward(self, x, state:list[torch.Tensor]):
            

            x = self.emb[x]

            
            torch.ops.rwkv.rwkvc(
                self.layers,
                self.dim,
                0,
                self.rx,
                x.clone().to(dtype=torch.float64, memory_format=torch.contiguous_format, device="cuda"),
                self.cudalnin,
                state[0],
                state[1],
                state[2],
                state[3],
                state[4],
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
            
            
            return self.buffer1, state
        
       
            
        
        
    if path.endswith(".cudarwkv"):
        myrwkv = torch.jit.load(path)
        returnObject: myRWKV = myrwkv
        return returnObject
    
    w = torch.load(path, map_location="cpu")
    # detach weights

    dims = len(w["blocks.0.att.key.weight"])
    layers = len(
        list(filter(lambda x: "blocks" in x and "ln1.bias" in x, w.keys())))

    myrwkv = myRWKV(w,dims,layers)
    del w
    torch.cuda.empty_cache()
    myrwkv.eval()
    

    if jit:
        myrwkv = torch.jit.script(myrwkv)
    if export:
        torch.jit.save(myrwkv, export+".cudarwkv")
        
        exit()
    returnObject: myRWKV = myrwkv
    return returnObject
