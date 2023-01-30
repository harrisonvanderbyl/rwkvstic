
from rwkvstic.rwkvMaster import RWKVMaster
from rwkvstic.agnostic.agnosticRwkv import AgnostigRWKV

from rwkvstic.agnostic.backends.torch import RWKVCudaQuantOps


def loadPreQuantized(path):
    import torch
    weights = torch.load(path)

    # filter out the keys that are not .block
    weightsKeys = [x for x in weights.keys() if "blocks" in x]
    n_layers = 0
    for weight in weightsKeys:
        ww = weight.split("blocks.")
        ww = ww[1].split(".")
        if int(ww[0]) > n_layers:
            n_layers = int(ww[0])

    ops = RWKVCudaQuantOps(
        preQuantized=True, embed=len(weights["blocks.0.ln2.weight"]), layers=(n_layers+1), chunksize=32, useGPU=torch.cuda.is_available(), runtimedtype=torch.bfloat16)
    for w in weights.keys():
        if "emb.weight" in w:
            weights[w] = ops.stack([ops.initCpuTensor(x.squeeze())
                                   for x in weights[w].split(1, 0)])
        else:
            weights[w] = ops.initTensor(weights[w])
    model = AgnostigRWKV(ops, weights)
    emptyState = ops.emptyState
    initTensor = ops.initTensor

    return RWKVMaster(model, emptyState, initTensor, ops.sample)
