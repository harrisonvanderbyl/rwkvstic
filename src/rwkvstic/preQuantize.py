
from rwkvstic.helpers.loadWeights import loadWeights
from rwkvstic.rwkvMaster import RWKVMaster
import torch
import gc
import inquirer
import os


# set torch threads to 8
torch.set_num_threads(8)


def preQuantized(path=None, chunksize=32) -> RWKVMaster:

    if (path == None):
        files = os.listdir()
        # filter by ending in .pth
        files = [f for f in files if f.endswith(
            ".pth")]

        questions = [
            inquirer.List('file',
                          message="What model do you want to quantize?",
                          choices=files,
                          )]
        path = inquirer.prompt(questions)["file"]

    mode = "pytorch-quant(gpu-8bit)"
    ops, weights = loadWeights(
        mode, path, runtimedtype=torch.bfloat16, chunksize=chunksize, useGPU=True, processEmb=True)

    gc.collect()
    torch.cuda.empty_cache()

    torch.save(weights, path.replace(".pth", ".pqth"))
