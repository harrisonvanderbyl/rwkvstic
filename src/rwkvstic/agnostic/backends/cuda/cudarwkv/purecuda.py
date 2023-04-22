import torch
from tqdm import tqdm

import os

current_path = os.path.dirname(os.path.abspath(__file__))


def OptRWKV(path, **kwargs):
    
    if path.endswith(".pth"):
        from rwkvstic.agnostic.backends.cuda.cudarwkv.export.export import OptRWKV as save
        return save(path)
    
    from torch.utils.cpp_extension import load
    import os
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    load(
        name=f"wkv_cuda",
        sources=[f"{current_path}/torchbind.cpp",
                f"{current_path}/rwkv.cu",
                ],
        )
    layers,embed = torch.ops.rwkv.load(path)
    print(layers,embed)
    

    class interop():
        def __init__(self):
            self.output = torch.zeros(50277,dtype=torch.float32)
            self.state = torch.ops.rwkv.attachState(self.output)
            self.emptyState = [
                torch.zeros(layers,embed,dtype=torch.float64)
            ]*5
            self.rnnOnly = True
            
        def forward(self, x, state:list[torch.Tensor]):
            for i,o in enumerate(state):
                # copy values in without changing the pointer
                self.state[i].copy_(o)
            
            torch.ops.rwkv.rwkvc(x[-1].item())
            torch.cuda.synchronize()
            
            return self.output, self.emptyState
        
    return interop()
