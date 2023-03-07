from rwkvstic.agnostic.backends.torch import RWKVCudaOps, RWKVPTTSExportOps, RWKVSplitCudaOps, RWKVCudaQuantOps, RWKVStreamBigOps, RWKVCudaDeepspeedOps, RWKVMpsOps, RWKVQuantMPSOps, RWKVStreamMPSOps
from rwkvstic.agnostic.backends.jax import RWKVJaxOps, RWKVNumpyOps
from rwkvstic.agnostic.backends.tensorflow import RWKVTFExport, RWKVTFOps
from rwkvstic.agnostic.backends.base import module
from rwkvstic.agnostic.backends.onnx import RWKVOnnxOps
from rwkvstic.agnostic.backends.coreml import RWKVCoreMLOps
from typing import Dict

Backends: Dict[str, module] = {
    "tensorflow(cpu/gpu)": RWKVTFOps,
    "pytorch(cpu/gpu)": RWKVCudaOps,
    "mps(mac/gpu)": RWKVMpsOps,
    "numpy(cpu)": RWKVNumpyOps,
    "jax(cpu/gpu/tpu)": RWKVJaxOps,
    "pytorch-deepspeed(gpu)": RWKVCudaDeepspeedOps,
    "pytorch-quant(gpu-8bit)": RWKVCudaQuantOps,
    "pytorch-stream(gpu-config-vram)": RWKVStreamBigOps,
    "pytorch-quant-mps(Apple,gpu-8bit)": RWKVQuantMPSOps,
    "pytorch-stream-mps(Apple,gpu-config-vram)": RWKVStreamMPSOps,
    "pytorch-split(2xgpu, broken, please use older version)": RWKVSplitCudaOps,
    "export-torchscript": RWKVPTTSExportOps,
    "export-tensorflow": RWKVTFExport,
    "onnx_export": RWKVOnnxOps,
    # "coreml": RWKVCoreMLOps
}


TORCH = "pytorch(cpu/gpu)"
JAX = "jax(cpu/gpu/tpu)"
TF = "tensorflow(cpu/gpu)"
TORCH_DEEPSPEED = "pytorch-deepspeed(gpu)"
TORCH_QUANT = "pytorch-quant(gpu-8bit)"
TORCH_STREAM = "pytorch-stream(gpu-config-vram)"
TORCH_QUANT_MPS = "pytorch-quant-mps(Apple,gpu-8bit)"
TORCH_STREAM_MPS = "pytorch-stream-mps(Apple,gpu-config-vram)"
# TORCH_SPLIT = "pytorch-split(2xgpu)" broken
TORCH_EXPORT = "export-torchscript"
TF_EXPORT = "export-tensorflow"
ONNX_EXPORT = "onnx_export"
MPS = "mps(mac/gpu)"
NUMPY = "numpy(cpu)"

# COREML = "coreml"
