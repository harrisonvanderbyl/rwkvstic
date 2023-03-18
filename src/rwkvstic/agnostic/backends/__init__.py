from rwkvstic.agnostic.backends.torch import RWKVCudaOps, RWKVPTTSExportOps, RWKVSplitCudaOps, RWKVCudaQuantOps, RWKVStreamBigOps, RWKVCudaDeepspeedOps, RWKVMpsOps, RWKVQuantMPSOps, RWKVStreamMPSOps
from rwkvstic.agnostic.backends.jax import RWKVJaxOps, RWKVNumpyOps, RWKVCuPyOps, RWKVCuPyQuantOps
from rwkvstic.agnostic.backends.tensorflow import RWKVTFExport, RWKVTFOps
from rwkvstic.agnostic.backends.base import module
from rwkvstic.agnostic.backends.onnx import RWKVOnnxOps
from rwkvstic.agnostic.backends.coreml import RWKVCoreMLOps
from typing import Dict

TORCH = "pytorch(cpu/gpu)"
JAX = "jax(cpu/gpu/tpu)"
FASTQUANT = "fastquant"
TF = "tensorflow(cpu/gpu)"
CUPY = "cupy(gpu)"
CUPY_QUANT = "cupy-quant(gpu)"
TORCH_DEEPSPEED = "pytorch-deepspeed(gpu)"
TORCH_QUANT = "pytorch-quant(gpu-8bit)"
TORCH_STREAM = "pytorch-stream(gpu-config-vram)"
TORCH_QUANT_MPS = "pytorch-quant-mps(Apple,gpu-8bit)"
TORCH_STREAM_MPS = "pytorch-stream-mps(Apple,gpu-config-vram)"
TORCH_SPLIT = "pytorch-split(multixgpu)"
TORCH_EXPORT = "export-torchscript"
TF_EXPORT = "export-tensorflow"
ONNX_EXPORT = "onnx_export"
MPS = "mps(mac/gpu)"
NUMPY = "numpy(cpu)"

Backends: Dict[str, module] = {
    (FASTQUANT): RWKVCudaOps,
    (TF): RWKVTFOps,
    (JAX): RWKVJaxOps,
    (TORCH): RWKVCudaOps,
    (TORCH_QUANT): RWKVCudaQuantOps,
    (TORCH_DEEPSPEED): RWKVCudaDeepspeedOps,
    (TORCH_STREAM): RWKVStreamBigOps,
    (TORCH_SPLIT): RWKVSplitCudaOps,
    (TORCH_EXPORT): RWKVPTTSExportOps,
    (TF_EXPORT): RWKVTFExport,
    (ONNX_EXPORT): RWKVOnnxOps,
    (MPS): RWKVMpsOps,
    (TORCH_QUANT_MPS): RWKVQuantMPSOps,
    (TORCH_STREAM_MPS): RWKVStreamMPSOps,
    (NUMPY): RWKVNumpyOps,
    (CUPY): RWKVCuPyOps,
    (CUPY_QUANT): RWKVCuPyQuantOps,
    # (COREML): RWKVCoreMLOps,

}


# COREML = "coreml"
