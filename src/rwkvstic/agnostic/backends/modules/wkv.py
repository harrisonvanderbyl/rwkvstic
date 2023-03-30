
from rwkvstic.agnostic.backends.modules.base import RwkvModule
import torch
class WKV(RwkvModule):
    def __init__(self):
        super(WKV, self).__init__()
        self.device = torch.device("cpu")
        self.custom = False
    def cudawkv(self, T: int, C: int, w, u, k, v, aa, bb, pp):
        assert 1 * C % min(C, 32) == 0
        k = k.to(torch.float64)
        w = w.contiguous().to(torch.float64)
        u = u.contiguous().to(torch.float64)
        k = k.contiguous().to(torch.float64)
        v = v.contiguous().to(torch.float64)
        y = torch.empty((T, C), device=self.device, memory_format=torch.contiguous_format, dtype=torch.float64)
        torch.ops.rwkv.wkv_forward(1, T, C, w, u, k, v, y, aa, bb, pp)
        return y.to(torch.float64), aa, bb, pp
    
    
    def wkv(self, T: int, C: int, w, u, k, v, aa, bb, pp):
        y = torch.empty((T, C), device=self.device, memory_format=torch.contiguous_format, dtype=torch.float64)
        for i in torch.arange(T):
            kk = torch.exp(k[i])
            vv = v[i]
            wr1 = aa +  torch.exp(u+w+k[i]) * vv
            wr2 = bb +  torch.exp(u+w+k[i])
            y[i] = wr1 / wr2
            aa = (aa + kk*vv) * torch.exp(w)
            bb = (bb + kk) * torch.exp(w)
        return y, aa, bb, pp

    def forward(self, T: int, C: int, w, u, k, v, aa, bb, pp):
        if self.custom == False:
            return self.wkv(T, C, w, u, k, v, aa, bb, pp)
        else:
            return self.cudawkv(T, C, w, u, k, v, aa, bb, pp)
        

    def config(self, **config):
        self.device = config["devices"][0]["device"]
        self.custom = config.get("custom", False)


