def loadCustomCudaModule():
        
    from torch.utils.cpp_extension import load
    import os
    current_path = os.path.dirname(os.path.abspath(__file__))

    load(
        name=f"wkv_cuda",
        sources=[f"{current_path}/wrapper.cpp",
                f"{current_path}/operators.cu",
                f"{current_path}/operators32.cu"
                ],
        verbose=False,
        extra_cuda_cflags=["-std=c++17", "-O3" ],
        
        is_python_module=False)