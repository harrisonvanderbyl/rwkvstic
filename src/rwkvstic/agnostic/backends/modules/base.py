import torch
class RwkvModule(torch.jit.ScriptModule):
    def __init__(self):
        super(RwkvModule, self).__init__()
        self.submodules = []
        self.subattributes = []
        

    def add_submodule(self, submodule):
        self.submodules.append(submodule)

    def to_gpu(self, device, maxVram):
        pass
