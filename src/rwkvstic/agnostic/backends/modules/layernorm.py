
from rwkvstic.agnostic.backends.modules.base import RwkvModule
import torch
class LayerNorm(RwkvModule):
    def __init__(self, weight, bias):
        super(LayerNorm, self).__init__()
        self.weight = weight.clone().to(torch.float64)
        self.bias = bias.clone().to(torch.float64)
        self.device = torch.device("cpu")
        self.runtimedtype = torch.float64
    def forward(self, x):

        x = x.to(self.device, dtype=self.runtimedtype)

        xee2 = x - torch.mean(x, dim=1, keepdim=True)

        x2 = torch.sqrt(torch.mean(xee2*xee2, dim=1, keepdim=True) +
                        1e-5)
        
        return self.weight*(xee2/x2) + self.bias

    def config(self, **config):
        self.device = config["devices"][0]["device"]
        self.runtimedtype = torch.float32 if self.device == "mps" else torch.float64
        self.weight = self.weight.to(self.device, dtype=self.runtimedtype)
        self.bias = self.bias.to(self.device, dtype=self.runtimedtype)

