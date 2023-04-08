
from rwkvstic.agnostic.backends.modules.base import RwkvModule
import torch
class CudaWKV(RwkvModule):
    def __init__(self):
        super(CudaWKV, self).__init__()
        self.y = torch.empty((1, 1), device="cpu", memory_format=torch.contiguous_format, dtype=torch.float64)
    def forward(self, T: int, C: int, w, u, k, v, aa, bb, pp):
        assert 1 * C % min(C, 32) == 0
        k = k.to(torch.float64)
        w = w.contiguous().to(torch.float64)
        u = u.contiguous().to(torch.float64)
        k = k.contiguous().to(torch.float64)
        v = v.contiguous().to(torch.float64)
        if self.y .shape[0] != T or self.y .shape[1] != C or self.y .device != w.device:
            self.y = torch.empty((T, C), device=w.device, memory_format=torch.contiguous_format, dtype=torch.float64)
        
        torch.ops.rwkv.wkv_forward(1, T, C, w, u, k, v, self.y, aa, bb, pp)
        return self.y.to(torch.float64), aa, bb, pp
    
    

class TorchWKV(RwkvModule):
    def __init__(self, device, dtype):
        super(TorchWKV, self).__init__()
        self.device = device
        self.runtimedtype = dtype
   
    def forward(self, T: int, C: int, w, u, k, v, aa, bb, pp):
        y = torch.empty((T, C), device=self.device, memory_format=torch.contiguous_format, dtype=self.runtimedtype)
        for i in torch.arange(T):
            kk = torch.exp(k[i])
            vv = v[i]
            wr1 = aa +  torch.exp(u+w+k[i]) * vv
            wr2 = bb +  torch.exp(u+w+k[i])
            y[i] = wr1 / wr2
            aa = (aa + kk*vv) * torch.exp(w)
            bb = (bb + kk) * torch.exp(w)
        return y, aa, bb, pp

   

        

class WKV(RwkvModule):
    def __init__(self):
        super(WKV, self).__init__()
        self.wkvmodule = CudaWKV()
   

    def forward(self, T: int, C: int, w, u, k, v, aa, bb, pp):
        return self.wkvmodule(T, C, w, u, k, v, aa, bb, pp)

    def config(self, **config):
        device = config["devices"][0]["device"]
        runtimedtype = torch.float32 if device == "mps" else torch.float64
        custom = config.get("custom", runtimedtype != torch.float32)

        if custom:
            self.wkvmodule = CudaWKV()

        else:
            self.wkvmodule = TorchWKV(device, runtimedtype)

