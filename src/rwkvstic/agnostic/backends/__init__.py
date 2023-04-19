from rwkvstic.agnostic.backends.torch import RWKVCudaOps, RWKVPTTSExportOps, RWKVCudaDeepspeedOps, RWKVMpsOps
from rwkvstic.agnostic.backends.jax import RWKVJaxOps, RWKVNumpyOps, RWKVCuPyOps, RWKVCuPyQuantOps
from rwkvstic.agnostic.backends.tensorflow import RWKVTFExport, RWKVTFOps
from rwkvstic.agnostic.backends.base import module
from rwkvstic.agnostic.backends.onnx import RWKVOnnxOps
from rwkvstic.agnostic.backends.coreml import RWKVCoreMLOps
from typing import Dict

TORCH = "pytorch(cpu/gpu)"
JAX = "jax(cpu/gpu/tpu)"
FASTQUANT = "fastquant"
FASTQUANTCUDA = "fastquant-cuda"
TF = "tensorflow(cpu/gpu)"
CUPY = "cupy(gpu)"
CUPY_QUANT = "cupy-quant(gpu)"
TORCH_DEEPSPEED = "pytorch-deepspeed(gpu)"
TORCH_EXPORT = "export-torchscript"
TF_EXPORT = "export-tensorflow"
ONNX_EXPORT = "onnx_export"
MPS = "mps(mac/gpu)"
NUMPY = "numpy(cpu)"

Backends: Dict[str, module] = {
    (FASTQUANT): RWKVCudaOps,
    (FASTQUANTCUDA): RWKVCudaOps,
    (TF): RWKVTFOps,
    (JAX): RWKVJaxOps,
    (TORCH): RWKVCudaOps,
    (TORCH_DEEPSPEED): RWKVCudaDeepspeedOps,
    (TORCH_EXPORT): RWKVPTTSExportOps,
    (TF_EXPORT): RWKVTFExport,
    (ONNX_EXPORT): RWKVOnnxOps,
    (MPS): RWKVMpsOps,
    (NUMPY): RWKVNumpyOps,
    (CUPY): RWKVCuPyOps,
    (CUPY_QUANT): RWKVCuPyQuantOps,
    # (COREML): RWKVCoreMLOps,

}


# COREML = "coreml"
