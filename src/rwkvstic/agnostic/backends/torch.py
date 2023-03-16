
from typing import List, Union
import rwkvstic.agnostic.backends.base as RWKVOp
import torch
from torch.utils.cpp_extension import load
import os
current_path = os.path.dirname(os.path.abspath(__file__))

# def test_cuda_mm8(b,t,c):
#     xx = torch.rand(b,t).to(dtype=torch.float16, device='cuda')
#     ww = (torch.rand(c,t)*255).to(dtype=torch.uint8, device='cuda').t()
    
#     yy = cuda_mm8(b,t,c,xx,ww)
#     yy1 = torch.mm(xx/255,ww.to(dtype=torch.float16))
#     print((yy - yy1).abs().max())

# test_cuda_mm8(5, 3, 3)
# test_cuda_mm8(5, 10, 3)
# test_cuda_mm8(5, 3, 10)
# test_cuda_mm8(5, 10, 10)
# test_cuda_mm8(5, 100, 10)
# test_cuda_mm8(5, 10, 100)
# test_cuda_mm8(5, 768, 768)
# test_cuda_mm8(5, 4*768, 768)
# test_cuda_mm8(5, 768, 4*768)
# exit()

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

            # return torch.nn.functional.layer_norm(x, x.shape, w.expand(x.shape), b.expand(x.shape))

        self.layernorm = ln
        self.emptyState = torch.tensor(
            self.emptyState, dtype=self.runtimedtype)
        self.stackEmb = True
        self.TensorType = torch.Tensor
        self.MatrixType = torch.Tensor
        self.VectorType = torch.Tensor
        def processEmbed(x): return x

        self.processEmbed = processEmbed


class RWKVCudaOps(RWKVPTOps):
    def __init__(self, layers, embed, *args, useGPU=None, runtimedtype=None, dev="cuda", **kwargs):
        super().__init__(layers, embed, *args, runtimedtype=runtimedtype, **kwargs)
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


class RWKVCudaQuantOps(RWKVPTOps):
    import torch
    


    def __init__(self, layers, embed, *args, runtimedtype=None, dtype=torch.float16, useGPU=None, chunksize=32, preQuantized=False, maxQuantTarget=None, target=None, dev="cuda", **kwargs):
        import torch
        import inquirer
        super().__init__(layers, embed, *args, dtype=dtype, **kwargs)
        load(
        name=f"wkv_cuda",
        sources=[f"{current_path}/cuda/wrapper.cpp",
                f"{current_path}/cuda/operators.cu"],
        verbose=True,
        extra_cuda_cflags=["-t=4","-std=c++17", "--use_fast_math",
                        "-O3", "--extra-device-vectorization"],
        is_python_module=False)

        @torch.jit.script
        def cuda_mm8(B: int, N: int, M: int, x, w, r):
            assert x.dtype == torch.float16
            assert w.dtype == torch.uint8


            # assert [x.shape[0], x.shape[1]] == [B, N]
            # assert [w.shape[0], w.shape[1]] == [N, M]
            # assert x.device == w.device
            # assert x.device.type == 'cuda'
            #print("cuda_mm8: ", B, N, M, x.device, w.device, x.dtype, w.dtype, x.shape, w.shape, x[0][0])
            # try:
            # print(x.shape, x.dtype)
            # print(B)
            if B > 1:
                return ((x*r) @ w.to(dtype=torch.float16)).squeeze()
                # too slow
                # use uint8@fp16 matmul library cutlass
        
            
            else:
                assert w.dtype == torch.uint8
                x = x[0]
                assert x.shape[0] == M
                w = w.contiguous()
                assert [w.shape[0],w.shape[1]] == [M, N]
                y = torch.zeros((N,), device=w.device, dtype=torch.float16)
                torch.ops.rwkv.mm8_one(M,N, x, w, y,r)
                
                y = y.to(dtype=torch.float16)

                # print(y.shape)
                return y.unsqueeze(0)
        # test
        # xx = torch.rand(1,5).half().cuda()
        # xy = torch.rand(10,5).mul(5).to(dtype=torch.uint8).cuda().t()
        # print(xx.unsqueeze(0)@xy.half())
        # tx = cuda_mm8(1,xy.shape[1],xy.shape[0],xx,xy)
        # print(tx.shape)
        # print(tx.cpu().float())
        
        # exit()
        @torch.jit.script
        def cuda_wkv(T: int, C: int, w, u, k, v, aa, bb, pp):
            assert 1 * C % min(C, 32) == 0
            assert k.dtype == torch.float16
            w = w.contiguous()
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            y = torch.empty((T, C), device="cuda", memory_format=torch.contiguous_format, dtype=torch.float16)
        
            torch.ops.rwkv.wkv_forward(1, T, C, w, u, k, v, y, aa, bb, pp)
            return y, aa, bb, pp
        
        def rwkvinterop(H, k, v, state, td,tf):
            
            rz, state[2], state[3], state[4] = cuda_wkv(H, embed, td.float(), tf.float(), k.half(), v.half(), state[2].float(), state[3].float(), state[4].float())
            return rz, state

        self.wkv = rwkvinterop

        def QuantizeMatrix(x, runtimeDtype, device, stream):
            rang = 255
            ran, mini = (x.max(0)[0]-x.min(0)[0])/rang,  x.min(0)[0]
            x = x.double()
            x = ((x-mini)/ran).round()
            if stream:
                x = x.to(
                    dtype=torch.uint8, non_blocking=True).pin_memory()
            else:
                x = x.to(
                    dtype=torch.uint8, non_blocking=True, device="cpu")

            return [x, ran.to(torch.float16).to(device=device), mini.to(runtimeDtype).to(device=device)]

        def QuantizedMatVec(x, y, runtimedtype):
            if len(x) != 3:
                return x.to(device=y.device, non_blocking=True).matmul(y.to(dtype=x.dtype).t()).to(dtype=y.dtype).t()
            rx, spread, zpoint = x
            yy = y

            yy = yy.to(dtype=dtype)

            # clone and reset stride to 1
            # yy = yy.clone().reshape(-1, yy.shape[1])

            # reset stride[1] to 1



            xmain = cuda_mm8(yy.shape[0], rx.shape[1], rx.shape[0], yy, rx.to(
                device=yy.device, non_blocking=True),spread).to(dtype=runtimedtype)
            

            # xmain2 = yy.matmul(
            #     rx.to(dtype=dtype, device=y.device, non_blocking=True)).to(dtype=runtimedtype)
            
            # print (xmain.shape, xmain2.shape)
            # print (xmain[0,0], xmain2[0,0])
            # print (xmain[0,1], xmain2[0,1])
            
            # print((xmain-xmain2).abs().max())
            zp = (y.to(runtimedtype)@zpoint).reshape(-1, 1)
            return xmain + zp
        cuda = dev
        dev = cuda if (inquirer.confirm(
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
                0)/1024/1024/1024 > target) if dev == cuda else False

            if preQuantized and len(x) == 3:
                return x[0].to(device=dev), x[1].to(dtype=runtimedtype, device=dev), x[2].to(dtype=runtimedtype, device=dev)

            if (len(x.shape) != 2):
                return x.to(dtype=runtimedtype, device=dev)

            if preQuantized:
                return x.to(dtype=dtype, device=dev)

            dontQuantize = (torch.cuda.max_memory_reserved(
                0)/1024/1024/1024 > maxQuantTarget) if dev == cuda else False

            if dontQuantize:
                if dostream:
                    return x.to(dtype=dtype, non_blocking=True).pin_memory()
                else:
                    return x.to(dtype=dtype, device=dev)

            splitmatrices = torch.chunk(x, chunksize, 1)

            xx = [QuantizeMatrix(x, runtimedtype, dev, dostream)
                  for x in splitmatrices]
            xxo = torch.cat([x[0]
                                          for x in xx], 1).cuda().t().contiguous()
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


class RWKVQuantMPSOps(RWKVCudaQuantOps):

    def __init__(self, layers, embed, *args, **kwargs):
        import torch
        super().__init__(layers, embed, *args, dev="mps", **kwargs)

        self.lerp = lambda x, y, z: x*(1-z)+y*z
        # roll not implemented in mps, makiing compatible
        self.roll = lambda x: x[(torch.arange(x.shape[0])-1).relu()]


class RWKVStreamMPSOps(RWKVQuantMPSOps):

    def __init__(self, layers, embed, *args, **kwargs):

        super().__init__(layers, embed,  *args, useGPU=True,
                         chunksize=1, maxQuantTarget=-1, **kwargs)


class RWKVSplitCudaOps(RWKVPTOps):
    import torch

    def __init__(self, layers, embed, *args, runtimedtype=torch.float32, dtype=torch.bfloat16, devices=None, **kwargs):
        super().__init__(layers, embed, *args, dtype=dtype, **kwargs)
        import inquirer
        import torch

        devices = inquirer.checkbox(
            'Which devices would you like to use?', choices=['cpu', *[f"cuda:{x}" for x in range(torch.cuda.device_count())]]) if devices is None else devices
        devices = sorted(devices)[::-1]

        self.initTensor = lambda x: x.to(dtype=runtimedtype).cuda() if len(
            x.shape) == 1 else list(map(lambda zx: zx[1].to(device=devices[zx[0]], dtype=torch.float32 if "cpu" in devices[zx[0]] else torch.bfloat16), enumerate(list(x.t().chunk(len(devices), dim=0)))))
        self.initCpuTensor = lambda x: x.to(dtype=runtimedtype)

        # for everything in self, if its a tensor, send to cuda
        # self.matvec = lambda x, y: x.mv(y.to(torch.float16)).to(runtimedtype)
        self.emptyState = self.emptyState.to(dtype=runtimedtype, device='cuda')

        self.minimum = torch.minimum

        def matvec(matx, y):
            chunks = list(map(lambda xx: xx[1].to(
                device=devices[xx[0]], dtype=matx[xx[0]].dtype, non_blocking=True), enumerate(y.chunk(len(devices), dim=1))))

            res = 0

            for i in range(0, len(chunks)):
                res = res + chunks[i].matmul(matx[i]).to(
                    dtype=runtimedtype, device=y.device, non_blocking=True)

            return res
        self.processEmbed = lambda x: x.to(device='cuda')

        self.matvec = matvec
