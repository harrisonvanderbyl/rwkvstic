from rwkvstic.rwkvMaster import RWKVMaster
from rwkvstic.agnostic.samplers.numpy import npsample


def initRWKVOriginal(path, strategy, tokenizer=None):
    from rwkv.model import RWKV
    modell = RWKV(path, strategy)

    class InterOp():
        RnnOnly = False

        def forward(self, x, y):

            return modell.forward(x, y)
    model = InterOp()
    emptyState = None
    import torch

    def initTensor(x): return torch.tensor(x)
    def intTensor(x): return [x] if type(x) == int else x
    return RWKVMaster(model, emptyState, initTensor, intTensor, npsample, tokenizer)
