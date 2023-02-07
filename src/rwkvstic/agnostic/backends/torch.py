
import rwkvstic.agnostic.backends.base as RWKVOp


class RWKVPTOps(RWKVOp.module):

    def __init__(self, layers, embed, dtype=None):
        import torch
        import inquirer
        super().__init__(layers, embed)
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
        self.initCpuTensor = lambda x: self.initTensor(x).cpu()
        self.klimit = torch.tensor(
            [18] * embed).to(dtype=self.dtype)
        self.minimum = torch.minimum
        self.sqrt = torch.sqrt
        self.mean = torch.mean
        self.relu = torch.relu
        self.stack = lambda x: x
        self.matvec = torch.mv
        # safe log
        self.log = lambda x: torch.complex(x, torch.zeros_like(x)).log()

        self.exp = lambda x: torch.exp(x).to(dtype=self.dtype)
        self.lerp = torch.lerp

        # module def
        self.module = torch.nn.Module

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x
        self.postProcessTensor = lambda x: x.float().cpu()

        # self.postProcessModule = ppm

        def layernorm(x, w, b) -> torch.Tensor:

            return torch.layer_norm(x, w.shape, w, b)
        self.layernorm = layernorm
        self.emptyState = torch.zeros(
            4*layers, embed, dtype=self.dtype)+0.0

        self.TensorType = torch.Tensor
        self.MatrixType = torch.Tensor
        self.VectorType = torch.Tensor


class RWKVPTCompatOps(RWKVPTOps):
    def __init__(self, layers, embed, *args):
        import torch
        RWKVPTOps.__init__(self, layers, embed, *args)
        self.relu = lambda x: torch.max(x, torch.zeros_like(x))
        self.matvec = lambda x, y: torch.sum(x*y, dim=1)

        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b

        self.layernorm = ln


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

        self.exp = lambda x: torch.exp(x).to(dtype=runtimedtype)

        self.initTensor = lambda x: x.to(dtype=self.dtype if len(
            x.shape) == 2 else runtimedtype, device='cuda')
        self.initCpuTensor = lambda x: x.to(dtype=runtimedtype)
        self.processEmbed = lambda x: x.cuda(non_blocking=True)
        self.klimit = self.klimit.to(dtype=runtimedtype, device='cuda')

        self.matvec = lambda x, y: x.mv(
            y.to(self.dtype)).to(runtimedtype)

        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b

        self.layernorm = ln

        self.emptyState = torch.zeros(
            4*layers, embed, dtype=runtimedtype, device="cuda")+0.01


class RWKVPTTSExportOps(RWKVCudaOps):
    def __init__(self, layers, embed, *args, includeSampler=None):
        super().__init__(layers, embed, *args)
        import torch
        import inquirer
        self.stack = lambda x: torch.stack(x)

        includeSampler = inquirer.confirm(
            "Include sampler?", default=True) if includeSampler is None else includeSampler

        if includeSampler:
            from rwkvstic.agnostic.samplers.torch import torchsample
            self.sample = torchsample
            self.postProcessTensor = lambda x: self.sample(
                x.float().cpu(), torch.tensor(1), torch.tensor(0.9))

        def exportTorchScript(x):
            torch.jit.save(torch.jit.trace(
                x, (torch.LongTensor([0]), self.emptyState), check_trace=False, strict=False), f"model-{layers}-{embed}-{'sampler' if includeSampler else 'logits'}-{'gpu' if self.useGPU else 'cpu'}-{self.dtype}.pt")
            exit()
        self.postProcessModule = exportTorchScript


class RWKVCudaDeepspeedOps(RWKVCudaOps):
    def __init__(self, layers, embed, *args):
        super().__init__(layers, embed, *args)

        try:
            import deepspeed
        except:
            raise ImportError("deepspeed not installed")

        self.postProcessModule = lambda x: deepspeed.init_inference(
            x, replace_method='auto', replace_with_kernel_inject=True).module


class RWKVCudaQuantOps(RWKVPTOps):
    def __init__(self, layers, embed, *args, runtimedtype=None, useGPU=None, chunksize=None, preQuantized=False, target=None):
        import torch
        import inquirer
        super().__init__(layers, embed, torch.bfloat16)

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
            rx, spread, zpoint = x
            y = y.reshape(rx.shape[0], -1)
            yy = y*spread

            rrx = rx.to(dtype=runtimedtype, device=y.device, non_blocking=True)

            xmain = rrx.matmul(yy.reshape(yy.shape[0], -1, 1)).sum(0).squeeze()

            return xmain + torch.tensordot(zpoint, y)
        dev = 'cuda' if (inquirer.confirm(
            "Use GPU?", default=True) if useGPU is None else useGPU) else 'cpu'

        # target is gb before cpu offload
        target = float(inquirer.text("Target size (in GB):",
                                     default="100")) if target is None and dev == "cuda" else target

        runtimedtype = inquirer.prompt([inquirer.List(
            'type',
            message="Dtype for operations:",
            choices=[torch.bfloat16, torch.float16, torch.float32, torch.float64])])['type'] if runtimedtype is None else runtimedtype

        chunksize = inquirer.prompt([inquirer.List(
            'chunksize',
            message="Chunksize(Trade speed for accuracy):",
            choices=[1, 2, 4, 8, 16, 32, 64, 128, 256])])['chunksize'] if chunksize is None else chunksize

        def initTensor(x):

            if preQuantized and len(x) == 3:
                return x[0].to(device=dev), x[1].to(dtype=runtimedtype, device=dev), x[2].to(dtype=runtimedtype, device=dev)

            if (len(x.shape) != 2):
                return x.to(dtype=runtimedtype, device=dev)

            splitmatrices = torch.chunk(x, chunksize, 1)
            dostream = (torch.cuda.max_memory_reserved(
                0)/1024/1024/1024 > target) if dev == "cuda" else False
            xx = [QuantizeMatrix(x, runtimedtype, dev, dostream)
                  for x in splitmatrices]
            xxo = torch.stack([x[0] for x in xx])
            xx1 = torch.stack([x[1] for x in xx])
            xx2 = torch.stack([x[2] for x in xx])
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

        self.klimit = self.klimit.to(dtype=runtimedtype, device=dev)

        self.emptyState = torch.zeros(
            4*layers, embed, dtype=runtimedtype, device=dev)+0.01


# class RWKVPoptorchOps(RWKVPTOps):
#     def __init__(self, layers, embed, *args):
#         super().__init__(layers, embed, *args)
#         try:
#             import poptorch
#         except:
#             raise ImportError("poptorch not installed")
#         self.postProcessModule = poptorch.inferenceModel


class RWKVStreamBigOps(RWKVPTOps):
    import torch

    def __init__(self, layers, embed, runtimedtype=torch.float32, dtype=torch.bfloat16, target=None, pinMem=None):
        import inquirer
        import torch
        super().__init__(layers, embed, dtype=dtype)

        pinMem = inquirer.prompt([inquirer.Confirm(
            'type',
            message=f"Pin memory to cpu?",
            default=True)])['type'] if pinMem is None else pinMem

        def pinmem(x):
            return x.pin_memory() if pinMem and x.device == "cpu" else x

        target = target if target is not None else float(
            input("Designate the amount of memory to allocate (in GB):"))
        self.initTensor = lambda x: pinmem(x.to(device='cpu' if len(x.shape) == 2 else "cuda", dtype=dtype if len(x.shape) == 2 else runtimedtype)) if (
            torch.cuda.max_memory_reserved(0)/1024/1024/1024) > target else x.to(dtype=dtype if len(x.shape) == 2 else runtimedtype).cuda()

        # for everything in self, if its a tensor, send to cuda

        self.initCpuTensor = self.initTensor
        self.klimit = self.klimit.cuda(non_blocking=True)
        self.matvec = lambda z, y: z.cuda(non_blocking=True).mv(
            y.to(dtype)).to(runtimedtype)
        self.emptyState = torch.zeros(
            4*layers, embed, dtype=runtimedtype, device="cuda")+0.01

        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b

        self.layernorm = ln


class RWKVSplitCudaOps(RWKVPTOps):
    import torch

    def __init__(self, layers, embed, runtimedtype=torch.float32, dtype=torch.bfloat16, target=None):
        super().__init__(layers, embed, dtype=dtype)
        import inquirer
        import torch

        devices = inquirer.checkbox(
            'Which devices would you like to use?', choices=['cpu', 'cuda:0', 'cuda:1'])

        self.initTensor = lambda x: x.to(dtype=runtimedtype).cuda() if len(
            x.shape) == 1 else list(map(lambda zx: zx[1].to(device=devices[zx[0]], dtype=torch.float32 if "cpu" in devices[zx[0]] else torch.bfloat16), enumerate(list(x.chunk(len(devices), dim=1)))))
        self.initCpuTensor = self.initTensor

        # for everything in self, if its a tensor, send to cuda
        # self.matvec = lambda x, y: x.mv(y.to(torch.float16)).to(runtimedtype)
        self.emptyState = torch.zeros(
            4*layers, embed, dtype=runtimedtype, device="cuda")+0.01

        self.minimum = lambda x, y: torch.min(x, torch.ones_like(x)*30)

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
