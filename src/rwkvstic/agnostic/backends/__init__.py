from rwkvstic.agnostic.backends.torch import RWKVCudaOps, RWKVPTCompatOps, RWKVPTTSExportOps, RWKVSplitCudaOps, RWKVCudaQuantOps, RWKVStreamBigOps, RWKVCudaDeepspeedOps
from rwkvstic.agnostic.backends.jax import RWKVJaxOps, RWKVNumpyOps
from rwkvstic.agnostic.backends.tensorflow import RWKVTFExport, RWKVTFOps
from rwkvstic.agnostic.backends.base import module
from typing import Dict
Backends: Dict[str, module] = {
    "tensorflow(cpu/gpu)": RWKVTFOps,
    "pytorch(cpu/gpu)": RWKVCudaOps,
    "numpy(cpu)": RWKVNumpyOps,
    "jax(cpu/gpu/tpu)": RWKVJaxOps,
    "pytorch-deepspeed(gpu)": RWKVCudaDeepspeedOps,
    "pytorch-quant(gpu-8bit)": RWKVCudaQuantOps,
    "pytorch-stream(gpu-config-vram)": RWKVStreamBigOps,
    "pytorch-split(2xgpu)": RWKVSplitCudaOps,
    "export-torchscript": RWKVPTTSExportOps,
    "export-tensorflow": RWKVTFExport,
    "pytorch-compatibility(cpu/debug)": RWKVPTCompatOps
}


TORCH = "pytorch(cpu/gpu)"
JAX = "jax(cpu/gpu/tpu)"
TF = "tensorflow(cpu/gpu)"
TORCH_DEEPSPEED = "pytorch-deepspeed(gpu)"
TORCH_QUANT = "pytorch-quant(gpu-8bit)"
TORCH_STREAM = "pytorch-stream(gpu-config-vram)"
TORCH_SPLIT = "pytorch-split(2xgpu)"
TORCH_EXPORT = "export-torchscript"
TF_EXPORT = "export-tensorflow"
TORCH_COMPAT = "pytorch-compatibility(cpu/debug)"
NUMPY = "numpy(cpu)"
