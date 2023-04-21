import torch
from tqdm import tqdm

import os

current_path = os.path.dirname(os.path.abspath(__file__))


def OptRWKV(path, jit=True, export=False,**kwargs):
    
    if export:
        from rwkvstic.agnostic.backends.cuda.cudarwkv.export import OptRWKV as save
        return save(path)
    
    from rwkvstic.agnostic.backends.cuda.cudarwkv.load import loadModule
    loadModule()
    layers,embed = torch.ops.rwkv.load(path)
    print(layers,embed)
    torch.ops.rwkv.toCuda()
    

    class interop():
        def __init__(self):
            self.emptyState = [
                torch.zeros(layers,embed,dtype=torch.float64).cuda().contiguous(),
                torch.zeros(layers,embed,dtype=torch.float64).cuda().contiguous(),
                torch.zeros(layers,embed,dtype=torch.float64).cuda().contiguous(),
                torch.zeros(layers,embed,dtype=torch.float64).cuda().contiguous(),
                torch.zeros(layers,embed,dtype=torch.float64).cuda().contiguous(),
            ]
            self.rnnOnly = True
            self.output = torch.zeros(50277,dtype=torch.float32).cuda()
        def forward(self, x, state:list[torch.Tensor]):

            torch.ops.rwkv.attachState(state[0],state[1],state[2],state[3],state[4])

            torch.ops.rwkv.rwkvc(x[-1].item())
            torch.ops.rwkv.getState(state[0],state[1],state[2],state[3],state[4], self.output)
            torch.cuda.synchronize()
            
            return self.output, state
        
    return interop()
