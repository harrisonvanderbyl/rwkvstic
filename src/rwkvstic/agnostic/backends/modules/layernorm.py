
from rwkvstic.agnostic.backends.modules.base import RwkvModule
import torch
class LayerNorm(RwkvModule):
    def __init__(self, weight, bias, device, dtype):
        super(LayerNorm, self).__init__()

        self.weight = weight.to(dtype).clone().to( device )
        self.bias = bias.to(dtype).clone().to( device )
        self.device = device
        self.subattributes = ["weight", "bias"]
    @ torch.jit.script_method
    def forward(self, x):
        xee2 = x - torch.mean(x, dim=1, keepdim=True)

        x2 = torch.sqrt(torch.mean(xee2*xee2, dim=1, keepdim=True) +
                        1e-5)
        o = self.weight*(xee2/x2) + self.bias

        return o

        # return torch.nn.functional.layer_norm(y, self.weight.shape, self.weight, self.bias, 1e-5
        #                                 )
    
