
from rwkvstic.helpers.loadWeights import loadWeights
from rwkvstic.rwkvMaster import RWKVMaster
import torch
import gc
import inquirer
import os


# set torch threads to 8
torch.set_num_threads(8)


def preQuantized(Path=None) -> RWKVMaster:

    if (Path == None):
        files = os.listdir()
        # filter by ending in .pth
        files = [f for f in files if f.endswith(
            ".pth")]

        questions = [
            inquirer.List('file',
                          message="What model do you want to quantize?",
                          choices=files,
                          )]
        Path = inquirer.prompt(questions)["file"]

    mode = "pytorch-quant(gpu-8bit)"
    ops, weights = loadWeights(
        mode, Path, runtimedtype=torch.bfloat16, chunksize=32, useGPU=False, processEmb=False)

    gc.collect()
    torch.cuda.empty_cache()

    torch.save(weights, Path.replace(".pth", ".pqth"))
