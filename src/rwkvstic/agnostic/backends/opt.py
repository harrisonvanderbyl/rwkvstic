import torch
from rwkvstic.agnostic.backends.modules.matmul import Linear
from rwkvstic.agnostic.backends.modules.layernorm import LayerNorm
from rwkvstic.agnostic.backends.modules.block import Block
from rwkvstic.agnostic.backends.modules.emb import RwkvEmb, RwkvModule
from tqdm import tqdm

import os

current_path = os.path.dirname(os.path.abspath(__file__))

method = torch.jit.script_method
module = torch.jit.ScriptModule
script = torch.jit.script
def OptRWKV(path, jit=True  , export=False, **kwargs):
    

    device = kwargs.get("device", "cuda")
    config = kwargs.get("config", {"devices":[{"device": "cpu", "dtype":torch.float32}]})
    if config["devices"][0]["device"] != "mps" and config["devices"][0]["device"] != "cpu":
        from rwkvstic.agnostic.backends.cuda.load import loadCustomCudaModule
        loadCustomCudaModule()
    else:
        print("Using CPU or MPS")
        # create dummy module for jit
        # add dummy to torch library
        
        def dummyfunc(i:int, T:int, C:int, w:torch.Tensor, u:torch.Tensor, k:torch.Tensor, v:torch.Tensor, y:torch.Tensor, aa:torch.Tensor, bb:torch.Tensor, pp:torch.Tensor):
                return torch.tensor(0)
            
 

        torch.ops.rwkv.wkv_forward = dummyfunc


    class myRWKV(RwkvModule):

        def __init__(self,w,dims,layers):
            super(myRWKV, self).__init__()
            print("Legacy RWKV")
            
            self.emptyState = torch.tensor(
                layers * [[[0]*dims]*4+[[-1e30]*dims]])

            self.emb =  RwkvEmb(w["emb.weight"])
            self.ln_out = LayerNorm(
                w["ln_out.weight"],
                w["ln_out.bias"]
            )

            self.ln_in = LayerNorm(
                w["blocks.0.ln0.weight"],
                w["blocks.0.ln0.bias"]
            )


            self.head = Linear(
                w["head.weight"]
            )

            
            
            print("Memory allocated before layers: ", torch.cuda.memory_allocated(device=device)/1024/1024, "MB")

            self.blocks = torch.nn.ModuleList([Block(w, i) for i in tqdm(range(layers), desc="loading layers")])

            del w

        
        def forward(self, x, state):
            
            x = self.emb(x)
            x = self.ln_in(x)

            for i, block in enumerate(self.blocks):

                x, rstate = block(x, state[i])
                state[i] = rstate

            x = self.ln_out(x)

            outx = self.head(x[-1:]).detach().squeeze()
            

            return outx, state
        
        def configs(self, **config):
            self.head.config(**config)
            self.emptyState = self.emptyState.to(config["devices"][0]["device"]).to(torch.float32 if config["devices"][0]["device"] == "mps" else torch.float64)
            self.ln_in.config(**config)
            self.ln_out.config(**config)
            for i, block in tqdm(enumerate(self.blocks),"configuring layers"):
                block.config(i,**config)

            
        
        
    if path.endswith(".rwkv"):
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
    
    
    myrwkv.configs(**config)

    # memory allocated after loading
    print("Memory allocated after layers: ", torch.cuda.memory_allocated(device=device)/1024/1024, "MB")
    
    
    if jit:
        myrwkv = torch.jit.script(myrwkv)
    if export:
        torch.jit.save(myrwkv, export+".rwkv")
        
        exit()
    returnObject: myRWKV = myrwkv
    return returnObject
