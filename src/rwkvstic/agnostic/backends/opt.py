import torch
from rwkvstic.agnostic.backends.modules.matmul import MM8
from rwkvstic.agnostic.backends.modules.layernorm import LayerNorm
from rwkvstic.agnostic.backends.modules.block import Block
from rwkvstic.agnostic.backends.modules.emb import RwkvEmb, RwkvModule
# set torch to use all 12 cores on amd
torch.set_num_threads(24)
# set torch to use hardware specific optimizations
torch.backends.cudnn.benchmark = True
# set torch to use deterministic algorithms
torch.backends.cudnn.deterministic = True
# set torch to use deterministic algorithms
torch.backends.cudnn.enabled = True
from torch.utils.cpp_extension import load
import gc
import os
current_path = os.path.dirname(os.path.abspath(__file__))

method = torch.jit.script_method
module = torch.jit.ScriptModule
script = torch.jit.script
with torch.no_grad():
    def OptRWKV(path, jit=True, export=False, maxvram=100, **kwargs):
      

        device = "cuda"

        dtype = torch.float32
        runtimedtype = torch.float32
        

        load(
            name=f"wkv_cuda",
            sources=[f"{current_path}/cuda/wrapper.cpp",
                    f"{current_path}/cuda/operators.cu"],
            verbose=False,
            extra_cuda_cflags=["-std=c++17", "-O3" ] + (["-gencode", "arch=compute_75,code=sm_75"] if torch.cuda.get_device_capability(0)[0] >= 7 else []) + (["-gencode", "arch=compute_80,code=sm_80"] if torch.cuda.get_device_capability(0)[0] >= 8 else []) + (["-gencode", "arch=compute_86,code=sm_86"] if torch.cuda.get_device_capability(0)[0] >= 8 else []),
            
            is_python_module=False)
        

        class myRWKV(RwkvModule):

            def __init__(self,w,dims,layers, device="cuda", maxvram = 100):
                super(myRWKV, self).__init__()
                print("Legacy RWKV")
                
                self.device = device
                self.dtype = dtype
                self.runtimedtype = runtimedtype

                

               
                # head = 50277
                
                self.emptyState = torch.tensor(
                    layers * [[[0]*dims]*4+[[-1e30]*dims]], device=device, dtype=runtimedtype)

                

                
               
                self.emb =  RwkvEmb(w["emb.weight"], device)
                self.ln_out = LayerNorm(
                    w["ln_out.weight"],
                    w["ln_out.bias"],
                    device=device
                )
                self.ln_in = LayerNorm(
                    w["blocks.0.ln0.weight"],
                    w["blocks.0.ln0.bias"],
                    device=device
                )

                self.head = MM8(
                    w["head.weight"],
                    device,
                    maxvram
                )
                
                print("Memory allocated before layers: ", torch.cuda.memory_allocated(device=device)/1024/1024, "MB")
                # loading bar
                from tqdm import tqdm
                self.blocks = torch.nn.ModuleList([Block(dims,w, i, device,maxvram) for i in tqdm(range(layers), desc="loading layers")])


                self.device = device

            
            
            @ method
            def forward(self, x, state):
                
                x = self.emb(x)
                x = self.ln_in(x)

                for i, block in enumerate(self.blocks):

                    x, rstate = block(x, state[i])
                    state[i] = rstate

                x = self.ln_out(x)

                outx = self.head(x[-1:]).detach().squeeze()
                

                return outx, state
            
            
        if path.endswith(".rwkv"):
            myrwkv = torch.jit.load(path)
            returnObject: myRWKV = myrwkv
            return returnObject
        w = torch.load(path, map_location="cpu")
        dims = len(w["blocks.0.att.key.weight"])
        layers = len(
            list(filter(lambda x: "blocks" in x and "ln1.bias" in x, w.keys())))

        myrwkv = myRWKV(w,dims,layers)
        del w
        myrwkv.eval()
        
        
        if jit:
            myrwkv = torch.jit.script(myrwkv)
        if export:
            torch.jit.save(myrwkv, export+".rwkv")
            
            exit()
        returnObject: myRWKV = myrwkv
        return returnObject
