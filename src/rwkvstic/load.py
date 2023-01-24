
import torch
from rwkvstic.helpers.loadWeights import loadWeights
from rwkvstic.agnostic.agnosticRwkv import AgnostigRWKV
from rwkvstic.agnostic.backends import Backends
from rwkvstic.interOpLoaders import tflite, torchscript
from rwkvstic.rwkvMaster import RWKVMaster
import torch
import gc
from typing import Tuple
import inquirer
import os

# set torch threads to 8
torch.set_num_threads(8)


def RWKV(Path=None, mode: Tuple[str, None] = None, *args, **kwargs) -> RWKVMaster:

    if (Path == None):
        files = os.listdir()
        # filter by ending in .pth
        files = [f for f in files if f.endswith(
            ".pth") or f.endswith(".pt") or f.endswith(".tflite")]

        questions = [
            inquirer.List('file',
                          message="What model do you want to use?",
                          choices=files,
                          )]
        Path = inquirer.prompt(questions)["file"]

    if Path.endswith(".pt"):
        return torchscript.initTorchScriptFile(Path)
    elif Path.endswith(".tflite"):
        return tflite.initTFLiteFile(Path)

    if mode is None:
        mode: str = inquirer.prompt([inquirer.List('mode',
                                                   message="What inference backend do you want to use?",
                                                   choices=Backends.keys(),
                                                   )])["mode"]

    ops, weights = loadWeights(mode, Path, *args, **kwargs)

    gc.collect()
    torch.cuda.empty_cache()

    model = AgnostigRWKV(ops, weights)
    emptyState = ops.emptyState
    initTensor = ops.initTensor

    ret = RWKVMaster(model, emptyState, initTensor, ops.sample)

    return ret
