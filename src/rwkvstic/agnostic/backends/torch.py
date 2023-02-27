
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
        # self.sample = torchsample

        def initTensor(x):
            result = x.to(dtype=self.dtype)

            return result

        self.initTensor = initTensor
        self.intTensor = lambda x: torch.tensor(x, dtype=torch.int64)
        self.initCpuTensor = lambda x: self.initTensor(x).cpu()
        self.klimit = torch.tensor(
            [18] * embed).to(dtype=self.dtype)
        self.maximum = torch.maximum
        self.minimum = torch.minimum
        self.unsqueeze = torch.unsqueeze
        self.expand = lambda x, y: x.expand(*y)
        self.pow = torch.pow
        self.add = torch.add
        self.one = torch.tensor(1, dtype=torch.int64)
        self.sqrt = torch.sqrt
        self.mean = torch.mean
        self.relu = torch.relu
        self.stack = torch.stack
        def roll(x): return torch.roll(x, x.shape[1])
        self.roll = roll
        def lenn(x): return x.shape[0]
        self.len = lenn
        self.cat = torch.cat
        def matvec(x, y): return torch.matmul(x, y.t()).t()
        self.matvec = matvec

        def prod(x): return torch.prod(x, dim=1)
        self.prod = prod
        # safe log
        def og(x): return torch.complex(x, torch.zeros_like(x)).log()
        self.log = og

        self.exp = torch.exp
        self.lerp = torch.lerp
        def rng(x: int): return torch.arange(x).to(dtype=torch.int64)
        self.rng = rng

        def emptyarray(x: int): return [torch.ones(0)]*x
        self.emptyarray = emptyarray

        def arrayPush(x: List[torch.Tensor], y, i: int):
            x[i] = y
            return x
        self.arrayPush = arrayPush

        self.mainarray = self.emptyarray
        self.pushstate = self.arrayPush

        def arrayGet(x: List[torch.Tensor], i: int): return x[i]
        self.arrayGet = arrayGet

        def lenn(x): return x.shape[0]
        self.len = lenn

        def pop(x): return x[-1]
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
            # nan on cuda
            # return torch.nn.functional.layer_norm(x, x.shape, w.expand(x.shape), b.expand(x.shape))

        self.layernorm = ln
        self.emptyState = torch.tensor(self.emptyState, dtype=self.dtype)

        self.TensorType = torch.Tensor
        self.MatrixType = torch.Tensor
        self.VectorType = torch.Tensor
        def processEmbed(x): return x

        self.processEmbed = processEmbed


class RWKVCudaOps(RWKVPTOps):
    def __init__(self, layers, embed, *args, useGPU=None, runtimedtype=None, **kwargs):
        super().__init__(layers, embed, *args, **kwargs)
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
            ret = x.to(dtype=self.dtype if len(x.shape) > 1 else runtimedtype).cuda(
                non_blocking=True)
            return ret

        self.initTensor = initTensor
        self.initCpuTensor = lambda x: x.to(dtype=self.runtimedtype)

        def processEmbed(x): return x.to(device='cuda')

        self.processEmbed = processEmbed

        def matvec(x, y): return x.matmul(
            y.to(dtype=x.dtype).t()).to(dtype=y.dtype).t()
        self.matvec = matvec

        self.emptyState = self.emptyState.to(dtype=runtimedtype, device='cuda')


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


class RWKVCudaQuantOps(RWKVPTOps):
    import torch

    def __init__(self, layers, embed, *args, runtimedtype=None, dtype=torch.bfloat16, useGPU=None, chunksize=32, preQuantized=False, maxQuantTarget=None, target=None, **kwargs):
        import torch
        import inquirer
        super().__init__(layers, embed, *args, dtype=dtype, **kwargs)

        def QuantizeMatrix(x, runtimeDtype, device, stream):
            rang = 255
            ran, mini = (x.max(0)[0]-x.min(0)[0])/rang,  x.min(0)[0]
            x = x.double()
            x = ((x-mini)/ran)
            if stream:
                x = x.to(
                    dtype=torch.uint8, non_blocking=True).pin_memory()
            else:
                x = x.to(
                    dtype=torch.uint8, non_blocking=True, device=device)

            return [x, ran.to(runtimeDtype).to(device=device), mini.to(runtimeDtype).to(device=device)]

        def QuantizedMatVec(x, y, runtimedtype):
            if len(x) != 3:
                return x.to(device=y.device, non_blocking=True).matmul(y.to(dtype=x.dtype).t()).to(dtype=y.dtype).t()
            rx, spread, zpoint = x
            yy = y*spread

            yy = yy.to(dtype=dtype)

            xmain = rx.to(dtype=dtype, device=y.device, non_blocking=True).matmul(
                yy.t()).to(dtype=runtimedtype).t()
            zp = (y@zpoint).reshape(-1, 1)
            return xmain + zp
        dev = 'cuda' if (inquirer.confirm(
            "Use GPU?", default=True) if useGPU is None else useGPU) else 'cpu'

        # target is gb before cpu offload
        target = float(inquirer.text("Target size (in GB):",
                                     default="100")) if target is None and dev == "cuda" else target

        maxQuantTarget = float(inquirer.text("Max quantization target size (in GB):",
                                             default="100")) if maxQuantTarget is None and dev == "cuda" else maxQuantTarget

        runtimedtype = inquirer.prompt([inquirer.List(
            'type',
            message="Dtype for operations:",
            choices=[torch.bfloat16, torch.float16, torch.float32, torch.float64])])['type'] if runtimedtype is None else runtimedtype

        chunksize = inquirer.prompt([inquirer.List(
            'chunksize',
            message="Chunksize(Trade speed for accuracy):",
            choices=[1, 2, 4, 8, 16, 32, 64, 128, 256])])['chunksize'] if chunksize is None else chunksize

        def initTensor(x):

            dostream = (torch.cuda.max_memory_reserved(
                0)/1024/1024/1024 > target) if dev == "cuda" else False

            if preQuantized and len(x) == 3:
                return x[0].to(device=dev), x[1].to(dtype=runtimedtype, device=dev), x[2].to(dtype=runtimedtype, device=dev)

            if (len(x.shape) != 2):
                return x.to(dtype=runtimedtype, device=dev)

            if preQuantized:
                return x.to(dtype=dtype, device=dev)

            dontQuantize = (torch.cuda.max_memory_reserved(
                0)/1024/1024/1024 > maxQuantTarget) if dev == "cuda" else False

            if dontQuantize:
                if dostream:
                    return x.to(dtype=dtype, non_blocking=True).pin_memory()
                else:
                    return x.to(dtype=dtype, device=dev)

            splitmatrices = torch.chunk(x, chunksize, 1)

            xx = [QuantizeMatrix(x, runtimedtype, dev, dostream)
                  for x in splitmatrices]
            xxo = torch.cat([x[0] for x in xx], 1)
            xx1 = torch.cat([x[1] for x in xx])
            xx2 = torch.cat([x[2] for x in xx])
            return xxo, xx1, xx2
        self.initTensor = initTensor
        self.stack = lambda x: torch.stack(
            x) if isinstance(x[0], torch.Tensor) else x
        self.initCpuTensor = lambda x: x.to(dtype=runtimedtype)
        self.processEmbed = lambda x: x.to(device=dev)

        self.postProcessModule = lambda x: x

        def matvec(x, y):
            return QuantizedMatVec(x, y, runtimedtype)

        self.matvec = matvec

        self.stackEmb = True

        self.klimit = self.klimit.to(dtype=runtimedtype, device=dev)

        self.emptyState = self.emptyState.to(dtype=runtimedtype, device=dev)


class RWKVStreamBigOps(RWKVCudaQuantOps):

    def __init__(self, layers, embed, *args, **kwargs):

        super().__init__(layers, embed, *args, useGPU=True,
                         chunksize=1, maxQuantTarget=-1, **kwargs)


class RWKVSplitCudaOps(RWKVPTOps):
    import torch

    def __init__(self, layers, embed, *args, runtimedtype=torch.float32, dtype=torch.bfloat16, devices=None, **kwargs):
        super().__init__(layers, embed, *args, dtype=dtype, **kwargs)
        import inquirer
        import torch

        devices = inquirer.checkbox(
            'Which devices would you like to use?', choices=['cpu', *[f"cuda:{x}" for x in range(torch.cuda.device_count())]]) if devices is None else devices

        self.initTensor = lambda x: x.to(dtype=runtimedtype).cuda() if len(
            x.shape) == 1 else list(map(lambda zx: zx[1].to(device=devices[zx[0]], dtype=torch.float32 if "cpu" in devices[zx[0]] else torch.bfloat16), enumerate(list(x.chunk(len(devices), dim=1)))))
        self.initCpuTensor = lambda x: x.to(dtype=runtimedtype)

        # for everything in self, if its a tensor, send to cuda
        # self.matvec = lambda x, y: x.mv(y.to(torch.float16)).to(runtimedtype)
        self.emptyState = self.emptyState.to(dtype=runtimedtype, device='cuda')

        self.minimum = torch.minimum

        def matvec(matx, y):
            chunks = list(map(lambda xx: xx[1].to(
                device=devices[xx[0]], dtype=matx[xx[0]].dtype, non_blocking=True), enumerate(y.chunk(len(devices), dim=0))))
            res = matx[0].mv(chunks[0]).to(
                dtype=runtimedtype, device=y.device, non_blocking=True)
            for i in range(1, len(chunks)):
                res = res + matx[i].mv(chunks[i]).to(
                    dtype=runtimedtype, device=y.device, non_blocking=True)

            return res

        self.stack = lambda x: x

        self.matvec = matvec
        self.layernorm = lambda x, w, b: torch.layer_norm(
            x.to(device=w.device), w.shape, w, b)
