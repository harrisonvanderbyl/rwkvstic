
from typing import List, Union
import rwkvstic.agnostic.backends.base as RWKVOp


class RWKVPTOps(RWKVOp.module):

    def __init__(self, layers, embed, *args, dtype=None, **kwargs):
        import torch
        import inquirer
        super().__init__(layers, embed,  dtype=dtype, *args, **kwargs)

        q = [inquirer.List(
            'type',
            message="Load model with which dtype?",
            choices=[torch.bfloat16, torch.float16, torch.float32, torch.float64])]

        if dtype is None:
            a = inquirer.prompt(q)
            dtype = a['type']
        self.dtype = dtype
        self.runtimedtype = kwargs.get('runtimedtype', dtype)
        # self.sample = torchsample

        def initTensor(x):
            if len(x.squeeze().shape) == 2:
                return x.to(self.dtype).t()
            else:
                return x.to(self.runtimedtype)


        self.initTensor = initTensor
        self.intTensor = lambda x: torch.tensor(
            x if isinstance(x, list) else [x], dtype=torch.int64)
        self.initCpuTensor = lambda x: self.initTensor(x).cpu()
        self.klimit = torch.tensor(
            [18] * embed).to(dtype=self.dtype)
        self.maximum = torch.maximum
        self.minimum = torch.minimum
        self.unsqueeze = torch.unsqueeze
        self.expand = lambda x, y: x.expand(*y)
        self.pow = torch.pow
        self.sqrt = torch.sqrt
        self.mean = torch.mean
        self.relu = torch.relu
        def tc(x): return torch.stack(x)
        self.stack = tc
        def tcs(x): return x
        self.mnstack = tcs
        def roll(x): return torch.roll(x, x.shape[1])
        self.roll = roll
        def lenn(x): return x.shape[0]
        self.len = lenn
        self.cat = torch.cat
        def matvec(x, y): return torch.matmul(y.to(x.dtype), x).to(y.dtype)
        self.matvec = matvec

        def prod(x): return torch.prod(x, dim=1)
        self.prod = prod
        # safe log
        def og(x): return torch.complex(x, torch.zeros_like(x)).log()
        self.log = og

        self.exp = torch.exp
        self.lerp = torch.lerp
        self.rng = torch.arange

        def emptyarray(x: int): return []
        self.emptyarray = emptyarray

        def arrayPush(x: List[torch.Tensor], y, i: int):
            return x + [y]
        self.arrayPush = arrayPush

        def arrayGet(x: List[torch.Tensor], i: int): return x[i]
        self.arrayGet = arrayGet

        def scatter(x, y, z):
            # like tensor[y] = z
            x[y] = z
            return x
        self.scatter = scatter

        self.scatterindices = [torch.tensor(
            [2, 3, 4]), torch.tensor([2, 3]), torch.tensor([0, 1])]

        def pop(x): return x[-1:]
        self.pop = pop

        # module def
        self.module = torch.nn.Module

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x
        self.logistical = torch.sigmoid
        def postProcessTensor(x): return x.float().cpu()
        self.postProcessTensor = postProcessTensor
        # self.postProcessModule = ppm

        def ln(x, w, b):
            # layernorm batchnorm
            xee2 = x - torch.mean(x, dim=1, keepdim=True)

            x2 = torch.sqrt(torch.mean(xee2*xee2, dim=1, keepdim=True) +
                            1e-5)
            o = w*(xee2/x2) + b

            return o

            # return torch.nn.functional.layer_norm(x, x.shape, w.expand(x.shape), b.expand(x.shape))

        self.layernorm = ln
        self.emptyState = torch.tensor(self.emptyState, dtype=self.dtype)
        self.stackEmb = True
        self.TensorType = torch.Tensor
        self.MatrixType = torch.Tensor
        self.VectorType = torch.Tensor
        def processEmbed(x): return x

        self.processEmbed = processEmbed


class RWKVCudaOps(RWKVPTOps):
    def __init__(self, layers, embed, *args, useGPU=None, runtimedtype=None, dev="cuda", **kwargs):
        super().__init__(layers, embed, *args,runtimedtype=runtimedtype, **kwargs)
        import inquirer
        import torch

        useGPU = inquirer.confirm(
            "Use GPU?", default=True) if useGPU is None else useGPU

        self.useGPU = useGPU

        if not useGPU:
            return

        runtimedtype = inquirer.prompt([inquirer.List(
            'type',
            message="Dtype for non-matrix ops:",
            choices=[torch.bfloat16, torch.float32, torch.float64])])['type'] if runtimedtype is None else runtimedtype

        self.runtimedtype = runtimedtype

        def initTensor(x):
            ret = x.to(dtype=self.dtype if len(x.shape) > 1 else runtimedtype).to(dev,
                                                                                  non_blocking=True)

            if len(ret.squeeze().shape) == 2:
                ret = ret.t()
            return ret

        self.initTensor = initTensor

        self.initCpuTensor = lambda x: x.to(dtype=self.runtimedtype)

        def processEmbed(x): return x.to(device=dev)

        self.processEmbed = processEmbed

        def matvec(x, y): return y.to(dtype=x.dtype).matmul(x
                                                            ).to(dtype=y.dtype)
        self.matvec = matvec

        self.emptyState = self.emptyState.to(dtype=runtimedtype, device=dev)


class RWKVMpsOps(RWKVCudaOps):

    def __init__(self, layers, embed, *args, **kwargs):
        super().__init__(layers, embed, dev="mps", *args, **kwargs)
        import torch
        self.lerp = lambda x, y, z: x + (y-x)*z
        self.roll = lambda x: x[(torch.arange(x.shape[0])-1).relu()]


class RWKVPTTSExportOps(RWKVCudaOps):
    def __init__(self, layers, embed, *args, includeSampler=None, **kwargs):
        super().__init__(layers, embed, *args, **kwargs)
        import torch
        import inquirer
        self.stack = torch.stack

        includeSampler = inquirer.confirm(
            "Include sampler?", default=True) if includeSampler is None else includeSampler

        if includeSampler:
            from rwkvstic.agnostic.samplers.torch import torchsample
            self.sample = torchsample
            self.postProcessTensor = lambda x: self.sample(
                x.float().cpu(), torch.tensor(1), torch.tensor(0.9))

        def exportTorchScript(x):
            xc = torch.jit.script(
                x)

            # save torchscript
            nm = f"model-{layers}-{embed}-{'sampler' if includeSampler else 'logits'}-{'gpu' if self.useGPU else 'cpu'}-{self.dtype}.pt"

            xc.save(nm)

            print(f"Saved model to {nm}")

            exit()
            # return xc
        self.postProcessModule = exportTorchScript


class RWKVCudaDeepspeedOps(RWKVCudaOps):
    def __init__(self, layers, embed, *args, **kwargs):
        super().__init__(layers, embed, *args, **kwargs)

        try:
            import deepspeed
        except:
            raise ImportError("deepspeed not installed")

        self.postProcessModule = lambda x: deepspeed.init_inference(
            x, replace_method='auto', replace_with_kernel_inject=True).module



