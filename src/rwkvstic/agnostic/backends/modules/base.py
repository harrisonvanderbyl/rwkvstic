import torch


class RwkvModule(torch.nn.Module):
    def __init__(self):
        super(RwkvModule, self).__init__()
        self.submodules = []
        self.subattributes = []
        

    def add_submodule(self, submodule):
        self.submodules.append(submodule)

    def config(self, config):
        pass
