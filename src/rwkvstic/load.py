
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


def RWKV(Path=None, mode: Tuple[str, None] = None, *args, **kwargs) -> RWKVMaster:

    if (Path == None):
        files = os.listdir()
        # filter by ending in .pth
        files = [f for f in files if f.endswith(
            ".pth") or f.endswith(".pt") or f.endswith(".tflite") or f.endswith(".pqth") or f.endswith(".jax.npy")]

        questions = [
            inquirer.List('file',
                          message="What model do you want to use?",
                          choices=files,
                          )]
        Path = inquirer.prompt(questions)["file"]
    else:
        if ("http" in Path):
            fileName = Path.split("/")[-1]
            if os.system("ls " + fileName):
                os.system(f"wget {Path}")
            Path = fileName

    if Path.endswith(".pt"):
        return torchscript.initTorchScriptFile(Path)
    elif Path.endswith(".tflite"):
        return tflite.initTFLiteFile(Path)
    elif Path.endswith(".pqth"):
        return prequantized.loadPreQuantized(Path)
    elif Path.endswith(".jax.npy"):
        return preJax.loadPreJax(Path)

    if mode is None:
        mode: str = inquirer.prompt([inquirer.List('mode',
                                                   message="What inference backend do you want to use?",
                                                   choices=Backends.keys(),
                                                   )])["mode"]

    ops, weights = loadWeights(mode, Path, *args, **kwargs)

    gc.collect()

    model = AgnostigRWKV(ops, weights)
    emptyState = ops.emptyState
    initTensor = ops.initTensor

    ret = RWKVMaster(model, emptyState, initTensor, ops.sample)

    return ret
