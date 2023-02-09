
from rwkvstic.helpers.loadWeights import loadWeights
from rwkvstic.agnostic.agnosticRwkv import AgnostigRWKV
from rwkvstic.agnostic.backends import Backends
from rwkvstic.interOpLoaders import tflite, torchscript, prequantized, preJax
from rwkvstic.rwkvMaster import RWKVMaster
import gc
from typing import Tuple
import inquirer
import os
# set torch threads to 8


def RWKV(path=None, mode: Tuple[str, None] = None, *args, tokenizer=None, **kwargs) -> RWKVMaster:

    if (path == None):
        files = os.listdir()
        # filter by ending in .pth
        files = [f for f in files if f.endswith(
            ".pth") or f.endswith(".pt") or f.endswith(".tflite") or f.endswith(".pqth") or f.endswith(".jax.npy")]

        questions = [
            inquirer.List('file',
                          message="What model do you want to use?",
                          choices=files,
                          )]
        path = inquirer.prompt(questions)["file"]
    else:
        if ("http" in path):
            fileName = path.split("/")[-1]
            if os.system("ls " + fileName):
                os.system(f"wget {path}")
            path = fileName

    # if (kwargs.get("legacy", None) is not None):
    #     from rwkvstic.interOpLoaders.legacy import RWKV_RNN
    #     return RWKV_RNN(path)

    if path.endswith(".pt"):
        return torchscript.initTorchScriptFile(path, tokenizer)
    elif path.endswith(".tflite"):
        return tflite.initTFLiteFile(path, tokenizer)
    elif path.endswith(".pqth"):
        return prequantized.loadPreQuantized(path, tokenizer)
    elif path.endswith(".jax.npy"):
        return preJax.loadPreJax(path, tokenizer)

    if mode is None:
        mode: str = inquirer.prompt([inquirer.List('mode',
                                                   message="What inference backend do you want to use?",
                                                   choices=Backends.keys(),
                                                   )])["mode"]

    ops, weights = loadWeights(mode, path, *args, **kwargs)

    gc.collect()

    model = AgnostigRWKV(ops, weights)
    emptyState = ops.emptyState
    initTensor = ops.initTensor

    ret = RWKVMaster(model, emptyState, initTensor,
                     ops.sample, tokenizer)

    return ret
