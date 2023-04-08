
from rwkvstic.agnostic.backends.modules.base import RwkvModule
import torch
class RwkvEmb (RwkvModule):
    def __init__(self,w):
        super().__init__()
        self.w = w.clone().cpu()
    def forward(self,x):
        return self.w[x.cpu()].to(torch.float64)
