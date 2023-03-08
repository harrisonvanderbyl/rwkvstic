
from rwkvstic.rwkvMaster import RWKVMaster
from rwkvstic.agnostic.agnosticRwkv import AgnosticRWKV

from rwkvstic.agnostic.backends.torch import RWKVCudaQuantOps


def loadPreQuantized(path, tokenizer=None):
    import torch

    weights = torch.load(
        path, **({"map_location": "cpu"} if not torch.cuda.is_available() else {}))

    # filter out the keys that are not .block
    weightsKeys = [x for x in weights.keys() if "blocks" in x]
    n_layers = 0
    for weight in weightsKeys:
        ww = weight.split("blocks.")
        ww = ww[1].split(".")
        if int(ww[0]) > n_layers:
            n_layers = int(ww[0])

    ops = RWKVCudaQuantOps(
        preQuantized=True, embed=len(weights["blocks.0.ln2.weight"]), layers=(n_layers+1), chunksize=32, target=100, maxQuantTarget=100, useLogFix="logfix" in path, useGPU=torch.cuda.is_available(), runtimedtype=torch.float32)
    model = AgnosticRWKV(ops, weights)
    emptyState = ops.emptyState
    initTensor = ops.initTensor
    intTensor = ops.intTensor
    sample = ops.sample

    return RWKVMaster(model, emptyState, initTensor, intTensor, sample, tokenizer)
