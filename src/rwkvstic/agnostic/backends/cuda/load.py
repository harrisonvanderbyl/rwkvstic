def loadCustomCudaModule():
        
    from torch.utils.cpp_extension import load
    import os
    current_path = os.path.dirname(os.path.abspath(__file__))

    load(
        name=f"wkv_cuda",
        sources=[f"{current_path}/wrapper.cpp",
                f"{current_path}/operators.cu",
                f"{current_path}/operators32.cu",
                f"{current_path}/cudarwkv.cu",
                ],
        )
