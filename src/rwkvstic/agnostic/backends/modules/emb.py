
from rwkvstic.agnostic.backends.modules.base import RwkvModule
import torch
class RwkvEmb (RwkvModule):
    def __init__(self,w,device,dtype):
        super().__init__()
        
        # self.device = device

        self.device = device
        self.w = w.clone().cpu()
        self.dtype = dtype
    def forward(self,x):
        return self.w[x.cpu()].to(
            device=self.device).to(self.dtype)
