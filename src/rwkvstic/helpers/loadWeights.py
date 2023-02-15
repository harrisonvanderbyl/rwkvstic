

from rwkvstic.agnostic.backends import Backends
from rwkvstic.agnostic.backends.base import module

from typing import Dict
from tqdm import tqdm
import inquirer


def loadWeights(mode, path, *args, processEmb=True, **kwargs):
    import torch
    n_layer = 0

    w: Dict[str, torch.Tensor] = torch.load(
        path, map_location="cpu")
    # refine weights
    keys = list(w.keys())
    for x in keys:
        w[x].requires_grad = False

        try:
            if (int(x.split('.')[1])+1 > n_layer):
                n_layer = int(x.split('.')[1])+1
        except:
            pass

    # store weights in self.w

    # Load Backend
    ops: module = Backends[mode](
        n_layer, len(w[f"blocks.0.ffn.time_mix_k"]), *args, **kwargs)

    if kwargs.get("lora_r", 0) > 0:
        keys = set(w.keys())
        for k in keys:
            k: str
            if k.endswith('.weight'):
                prefix = k[:-len('.weight')]
                lora_A = prefix + '.lora_A.weight'
                lora_B = prefix + '.lora_B.weight'
                if lora_A in keys:
                    assert lora_B in keys
                    print(f'merging {lora_A} and {lora_B} into {k}')
                    assert w[lora_B].shape[1] == w[lora_A].shape[0] == kwargs["lora_r"]
                    w[k] += w[lora_B] @ w[lora_A] * \
                        (kwargs["lora_alpha"] / kwargs["lora_r"])
                    del w[lora_A]
                    del w[lora_B]
        if kwargs.get('lora_bake', None) is None:
            kwargs['lora_bake'] = inquirer.confirm(
                'bake lora into weights?')
        if kwargs['lora_bake']:
            print('baking lora into weights')
            torch.save(w, path.replace('.pth', '.Baked.pth'))

    keys = list(w.keys())
    for x in keys:

        if '.time_' in x:
            w[x] = w[x].squeeze()

        if '.time_decay' in x:
            w[x] = -torch.exp(w[x].double())
            if not ops.useLogFix:
                w[x] = torch.exp(w[x])

        if 'receptance.weight' in x:
            w[x] = -w[x]

    # Transform Weights from backend
    for x in tqdm(list(w.keys())):
        if "emb.weight" in x:
            if (processEmb):
                w[x] = ops.stack(list(map(lambda rrx: ops.initCpuTensor(
                    rrx.squeeze()), w[x].split(1, 0))))
        else:
            w[x] = ops.initTensor(w[x])

    return ops, w
