
from rwkvstic.rwkvMaster import RWKVMaster
from rwkvstic.agnostic.agnosticRwkv import AgnostigRWKV

from rwkvstic.agnostic.backends.jax import RWKVJaxOps


def loadPreJax(path, tokenizer=None, end_adj=0.0):
    import jax
    weights = jax.numpy.load(path, allow_pickle=True)
    # filter out the keys that are not .block
    weightsKeys = [x for x in weights.keys() if "blocks" in x]
    n_layers = 0
    for weight in weightsKeys:
        ww = weight.split("blocks.")
        ww = ww[1].split(".")
        if int(ww[0]) > n_layers:
            n_layers = int(ww[0])

    ops = RWKVJaxOps(
        embed=len(weights["blocks.0.ln2.weight"]), layers=n_layers, preJax=True)
    for w in weights.keys():
        if "emb" in w:
            weights[w] = ops.stack([ops.initTensor(x) for x in weights[w]])
            continue

        weights[w] = ops.initTensor(weights[w])

    model = AgnostigRWKV(ops, weights)
    emptyState = ops.emptyState
    initTensor = ops.initTensor

    return RWKVMaster(model, emptyState, initTensor, ops.sample, tokenizer, end_adj=end_adj)
