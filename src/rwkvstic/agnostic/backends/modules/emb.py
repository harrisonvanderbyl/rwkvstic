
from rwkvstic.agnostic.backends.modules.base import RwkvModule
import torch
class RwkvEmb (RwkvModule):
    def __init__(self,w,device) -> None:
        super().__init__()
        
        # self.device = device
        self.device = device
        self.w = w.clone().cpu()
    def forward(self,x):
        return self.w[x.cpu()].float().to(
            device=self.device)
