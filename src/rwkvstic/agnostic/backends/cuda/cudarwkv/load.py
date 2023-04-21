from torch.utils.cpp_extension import load
import os
current_path = os.path.dirname(os.path.abspath(__file__))
def loadModule():
    load(
        name=f"wkv_cuda",
        sources=[f"{current_path}/rwkv.cpp",
                 
                f"{current_path}/rwkv.cu",
                ],
        )