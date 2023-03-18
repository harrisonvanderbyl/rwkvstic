
from rwkvstic.agnostic.backends.modules.base import RwkvModule
import torch
class LayerNorm(RwkvModule):
    def __init__(self, weight, bias, device):
        super(LayerNorm, self).__init__()

        self.weight = weight.float().clone().to( device )
        self.bias = bias.float().clone().to( device )
        self.subattributes = ["weight", "bias"]
    @ torch.jit.script_method
    def forward(self, y):

        return torch.nn.functional.layer_norm(y, self.weight.shape, self.weight, self.bias, 1e-5
                                        )
    
