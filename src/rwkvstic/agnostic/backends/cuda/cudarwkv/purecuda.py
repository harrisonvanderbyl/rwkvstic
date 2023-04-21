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
            # state[0] *= 0
            # state[1] *= 0
            # state[2] *= 0
            # state[3] *= 0
            # state[4] *= 0
            # self.output *= 0
            # torch.cuda.synchronize()
            torch.ops.rwkv.attachState(state[0],state[1],state[2],state[3],state[4])
            # print(x[-1].item())
            torch.ops.rwkv.rwkvc(0)
            torch.ops.rwkv.getState(state[0],state[1],state[2],state[3],state[4], self.output)
            torch.cuda.synchronize()
            # print(self.output)
            return self.output, state
        
    return interop()
    # class myRWKV(torch.nn.Module):
              

            
        # def forward(self, x, state:list[torch.Tensor]):
            


            
        #     torch.ops.rwkv.save(
        #         self.layers,
        #         self.dim,
        #         x,
        #         self.rx,
        #         self.cudalnin,
        #         state[0],
        #         state[1],
        #         state[2],
        #         state[3],
        #         state[4],
        #         self.buffer0,
        #         self.buffer1,
        #         self.buffer2,
        #         self.buffer3,
        #         self.mixk,
        #         self.mixv,
        #         self.mixr,
        #         self.attkeyweights,
        #         self.attvalueweights,
        #         self.attreceptanceweights,
        #         self.attkeyranges,
        #         self.attvalueranges,
        #         self.attreceptanceranges,
        #         self.attkeyzp,
        #         self.attvaluezp,
        #         self.attreceptancezp,
        #         self.attoutputweights,
        #         self.attoutputranges,
        #         self.attoutputzp,
        #         self.mixffnk,
        #         self.mixffnr,
        #         self.ffnkeyweights,
        #         self.ffnvalueweights,
        #         self.ffnreceptanceweights,
        #         self.ffnkeyranges,
        #         self.ffnvalueranges,
        #         self.ffnreceptanceranges,
        #         self.ffnkeyzp,
        #         self.ffnvaluezp,
        #         self.ffnreceptancezp,
        #         self.ffnkbuf,
        #         self.ffnvbuf,
        #         self.ffkeybuffer,
        #         self.decay,
        #         self.bonus,
        #         self.cudahead,
        #         self.cudaheadr,
        #         self.cudaheadzp
        #     )

            
            
            
        #     return self.buffer1, state
        
       
            
        
        
    

    # myrwkv = myRWKV(path)
    
    # myrwkv.eval()
    

   
    # returnObject: myRWKV = myrwkv
    # return returnObject
