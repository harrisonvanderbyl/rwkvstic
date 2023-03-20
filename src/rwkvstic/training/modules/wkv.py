import torch

class wkv_power(torch.nn.Module):
    def __init__(self, dims, T):
        super(wkv_power, self).__init__()
        self.time_first = torch.nn.Parameter(torch.randn(dims))
        self.time_decay = torch.nn.Parameter(torch.randn(dims))
        
        self.T = T

        self.register_parameter("time_first", self.time_first)
        self.register_parameter("time_decay", self.time_decay)
        self.register_buffer("mask", torch.ones(T, T).tril().unsqueeze(-1).to(torch.bool), persistent=False)
        self.register_buffer("tri", ((torch.arange(T).expand(T, T)+1).t() -
            torch.arange(T)).tril().unsqueeze(-1), persistent=False)
        
        # set all weights to 0
        

        

    def forward(self, k,v):
        
        vx = v * k.exp()
        kx = k.exp()

        t = ((self.time_decay.expand(self.T,self.T,-1)*self.tri).exp()*self.mask)
        

        kxr = (kx*t).sum(1)
        vxr = (vx*t).sum(1)
 
        vxm = vx * self.time_first.exp()
        kxm = kx * self.time_first.exp()
        vxx = vxm + kxr
        kxx = kxm + vxr
        wkv = vxx/kxx
        return wkv