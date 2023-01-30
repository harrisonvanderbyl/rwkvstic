
from rwkvstic.helpers.loadWeights import loadWeights
from rwkvstic.rwkvMaster import RWKVMaster
import torch
import gc
import inquirer
import os


# set torch threads to 8
torch.set_num_threads(8)


def preJax(path=None) -> RWKVMaster:

    if (path == None):
        files = os.listdir()
        # filter by ending in .pth
        files = [f for f in files if f.endswith(
            ".pth")]

        questions = [
            inquirer.List('file',
                          message="What model do you want to convert to jax?",
                          choices=files,
                          )]
        path = inquirer.prompt(questions)["file"]

    mode = "jax(cpu/gpu/tpu)"
    ops, weights = loadWeights(
        mode, path)

    gc.collect()
    torch.cuda.empty_cache()
    import jax

    # save
    jax.numpy.save(path.replace(".pth", ".jax.npy"), weights)
