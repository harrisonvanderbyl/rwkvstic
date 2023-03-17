import torch
# set torch to use all 12 cores on amd
torch.set_num_threads(24)
# set torch to use hardware specific optimizations
torch.backends.cudnn.benchmark = True
# set torch to use deterministic algorithms
torch.backends.cudnn.deterministic = True
# set torch to use deterministic algorithms
torch.backends.cudnn.enabled = True
from torch.utils.cpp_extension import load
import gc
import os
current_path = os.path.dirname(os.path.abspath(__file__))

# method = torch.jit.script_method
# module = torch.jit.ScriptModule
# script = torch.jit.script
method = lambda x:x #.script_method
module = torch.nn.Module #.ScriptModule
script = lambda x:x #.script
with torch.no_grad():
    def OptRWKV(path):

        device = "cuda"

        dtype = torch.float16
        runtimedtype = torch.float32
        # @script
        # def powerTri(t, p:int):
        #     t = t.expand(p, p, -1)

        #     tri = ((torch.arange(p,device="cuda").expand(p, p)+1).t() -
        #         torch.arange(p,device="cuda")).tril().unsqueeze(-1)

        #     mask = torch.ones(p, p,device="cuda").tril().unsqueeze(-1).to(torch.bool)

        #     return ((t*tri).exp()*mask)
        
        load(
            name=f"wkv_cuda",
            sources=[f"{current_path}/cuda/wrapper.cpp",
                    f"{current_path}/cuda/operators.cu"],
            verbose=True,
            extra_cuda_cflags=["-std=c++17", "-O3"] + (["-gencode", "arch=compute_75,code=sm_75"] if torch.cuda.get_device_capability(0)[0] >= 7 else []) + (["-gencode", "arch=compute_80,code=sm_80"] if torch.cuda.get_device_capability(0)[0] >= 8 else []) + (["-gencode", "arch=compute_86,code=sm_86"] if torch.cuda.get_device_capability(0)[0] >= 8 else []),

            is_python_module=False)
        
        
        # def rwkvinterop(H, k, v, state, td,tf):
            
        #     rz, state[2], state[3], state[4] = cuda_wkv(H, embed, td.float(), tf.float(), k.half(), v.half(), state[2].float(), state[3].float(), state[4].float())
        #     return rz, state
        
        class MM8(module):
            def __init__(self, weight, runtimedtype):
                
                super(MM8, self).__init__()

                splitmatrices = torch.chunk(weight, 32, 1)

                self.runtimedtype = runtimedtype

                xx = [self.QuantizeMatrix(x)
                    for x in splitmatrices]
                self.weight = (torch.cat([x[0] for x in xx], 1).t().contiguous()).to (
                    device=device)
                self.M = self.weight.shape[0]
                self.N = self.weight.shape[1]
                self.range = (torch.cat([x[1] for x in xx])).to (
                    device=device)
                self.offset = (torch.cat([x[2] for x in xx])).to (
                    device=device)
                
            
            def QuantizeMatrix(self, x):
                rang = 255
                ran, mini = (x.max(0)[0]-x.min(0)[0])/rang,  x.min(0)[0]
                x = x.double()
                x = ((x-mini)/ran).round()
                x = x.to(
                        dtype=torch.uint8)
                return [x, ran.to(torch.float16), mini.to(self.runtimedtype)]
            @ method
            def cuda_mm8(self, N:int, M:int, x, w, r):
                # assert B == 1
                # assert w.dtype == torch.uint8
                x = x[0]
                # assert x.shape == [M]
                
                # assert w.shape == [M, N]
                y = torch.zeros(N, device=w.device, dtype=torch.float16)
                torch.ops.rwkv.mm8_one(M,N, x, w, y,r)
                
                y = y.to(dtype=torch.float16)

                return y.unsqueeze(0)
            @ method
            def cuda_mm8_seq(self,B:int, N:int, M:int, x, w, r):
                # assert B == 1
                # assert w.dtype == torch.uint8
              
                # assert x.shape == [M]
                
                # assert w.shape == [M, N]
                y = torch.zeros(B,N, device=w.device, dtype=torch.float16)
                torch.ops.rwkv.mm8_seq(B,M,N, x, w, y,r)
                
                y = y.to(dtype=torch.float16)

                return y
            @ method
            def forward(self, y):
                
                B = y.shape[0]
                yy = y.to(torch.float16)
                if B > 1:
                    xmain = (yy*self.range) @ self.weight.to(torch.float16)
                    # print(y.shape)
                    # print(self.offset.shape)
                    # print(y.dtype, y.device)
                    # print(self.offset.dtype, self.offset.device)
                    
                    zp = (y.mv(self.offset)).reshape(-1, 1)
                    
                     
                    return xmain + zp 
                
                xmain = self.cuda_mm8(self.N, self.M, yy, self.weight,self.range).to(self.runtimedtype)
                

                zp = (y@self.offset).to(self.runtimedtype)
                return (xmain + zp)
        class LayerNorm(module):
            def __init__(self, weight, bias):
                super(LayerNorm, self).__init__()

                self.weight = weight.to(runtimedtype).to (
                    device=device)
                self.bias = bias.to(runtimedtype).to (
                    device=device)
            @ method
            def forward(self, y):

                return torch.nn.functional.layer_norm(y, self.weight.shape, self.weight, self.bias, 1e-5
                                             )
        class Block(module):
            def __init__(self, dims, w, i):
                super(Block, self).__init__()

                self.dtype = dtype
                self.runtimedtype = runtimedtype
                self.t = []

                
                self.ln1 = LayerNorm(
                    w[f"blocks.{i}.ln1.weight"], w[f"blocks.{i}.ln1.bias"])
                self.ln2 = LayerNorm(
                    w[f"blocks.{i}.ln2.weight"], w[f"blocks.{i}.ln2.bias"])
               
                
                self.attkey = MM8(
                    w[f"blocks.{i}.att.key.weight"], runtimedtype)
                self.attvalue = MM8(
                    w[f"blocks.{i}.att.value.weight"], runtimedtype)
                self.ffnkey = MM8(
                    w[f"blocks.{i}.ffn.key.weight"], runtimedtype)
                self.ffnvalue = MM8(
                    w[f"blocks.{i}.ffn.value.weight"], runtimedtype)
                self.attout = MM8(
                    w[f"blocks.{i}.att.output.weight"], runtimedtype)
                self.ffnreceptance = MM8(
                    w[f"blocks.{i}.ffn.receptance.weight"], runtimedtype)
                self.attreceptance = MM8(
                    w[f"blocks.{i}.att.receptance.weight"], runtimedtype)

                self.attmix= torch.stack((w[f"blocks.{i}.att.time_mix_k"].squeeze(),
                  w[f"blocks.{i}.att.time_mix_v"].squeeze(),
                     w[f"blocks.{i}.att.time_mix_r"].squeeze())).to(runtimedtype).to (
                    device=device).unsqueeze(1)
                
                self.time_first = w[f"blocks.{i}.att.time_first"].squeeze().to(runtimedtype).to (
                    device=device)

                self.time_decay = w[f"blocks.{i}.att.time_decay"].squeeze().to(runtimedtype).to (
                    device=device).double().exp().neg()

                # self.t = [powerTri(self.time_decay, i) for i in range(1, 21)]

                # self.ffntime_mix_k = (
                #     w[f"blocks.{i}.ffn.time_mix_k"].squeeze().to(runtimedtype).to (
                #     device=device))
                # self.ffntime_mix_r = (
                #     w[f"blocks.{i}.ffn.time_mix_r"].squeeze().to(runtimedtype).to (
                #     device=device))
                self.ffnmix= torch.stack((w[f"blocks.{i}.ffn.time_mix_k"].squeeze(),
                    w[f"blocks.{i}.ffn.time_mix_r"].squeeze())).to(runtimedtype).to (
                    device=device).unsqueeze(1)
                self.attreceptance = MM8(
                    w[f"blocks.{i}.att.receptance.weight"], runtimedtype)
            @method
            def cuda_wkv(self, T: int, C: int, w, u, k, v, aa, bb, pp):
                assert 1 * C % min(C, 32) == 0
                assert k.dtype == torch.float16
                w = w.contiguous()
                u = u.contiguous()
                k = k.contiguous()
                v = v.contiguous()
                y = torch.empty((T, C), device="cuda", memory_format=torch.contiguous_format, dtype=torch.float16)
            
                torch.ops.rwkv.wkv_forward(1, T, C, w, u, k, v, y, aa, bb, pp)
                return y.to(self.runtimedtype), aa, bb, pp
            @ method
            def forward(self, x, state):

                xy = self.ln1(x)

                tc = xy.roll(1, 0)
                rmc = tc[0].clone()
                tc[0] = state[0]
                state[0] = rmc

                mix = torch.lerp(tc.unsqueeze(0), xy.unsqueeze(0), self.attmix)

                k = self.attkey(mix[0])

                # mix = torch.lerp(tc, xy, self.atttmixemixix_v)

                v = self.attvalue(mix[1])

                # mix = torch.lerp(tc, xy, self.atttmixemixix_r)

                r = self.attreceptance(mix[2]).sigmoid()

                # WKV kernel original

                wkv, state[2],state[3],state[4] = self.cuda_wkv(k.shape[0], k.shape[1], self.time_decay.float(), self.time_first.float(), k.half(), v.half(), state[2].float(), state[3].float(), state[4].float())
                wkv *= r
                # WKV kernel experimental
                # vx_kx = (k).exp().unsqueeze(0) .expand(
                #     2, k.shape[0], k.shape[1]).clone()
                # vx_kx[0] *= v

                # t = powerTri(self.time_decay,k.shape[0])
                # vx_kx[0][0] += state[2]
                # vx_kx[1][0] += state[3]

                # rza = torch.einsum("rki,jki->rji", vx_kx, t)

                # vx_kx *= self.time_first.exp()
                # vx_kx += rza
                # vx_kx[0] = r*vx_kx[0]
                # vx_kx[1] = 1/vx_kx[1]
                # wkv = vx_kx.prod(0)

                # state[2] = rza[0][-1]
                # state[3] = rza[1][-1]

                rz = x + self.attout(wkv)

                ddd = self.ln2(rz)

                rc = ddd.roll(1, 0)
                dc = rc[0].clone()
                rc[0] = state[1]
                state[1] = dc

                fmix = torch.lerp(rc, ddd, self.ffnmix)

                kf = self.ffnkey(fmix[0]).relu()

                rf = self.ffnreceptance(fmix[1]).to(
                    self.runtimedtype).sigmoid()

                rvm = self.ffnvalue(torch.square(kf))

                out = rvm * rf + rz

                return out, state

                # stuff
        class RwkvEmb (torch.nn.Module):
            def __init__(self,w) -> None:
                super().__init__()
                self.w = w.to (
                    device=device)
                self.device = device
            def forward(self,x):
                return self.w[x.to(
                    device=self.device)]
        class myRWKV(torch.nn.Module):

            def __init__(self,path):
                super(myRWKV, self).__init__()
                print("Legacy RWKV")
                
                    
                self.dtype = dtype
                self.runtimedtype = runtimedtype

                w = torch.load(path, map_location="cpu")

                dims = w["blocks.0.att.key.weight"].shape[0]
                # head = 50277
                layers = len(
                    list(filter(lambda x: "blocks" in x and "ln1.bias" in x, w.keys())))

                self.emptyState = torch.tensor(
                    layers * [[[0]*dims]*4+[[-1e30]*dims]], device=device, dtype=runtimedtype)

                

                
               
                self.emb =  RwkvEmb(w["emb.weight"].to(runtimedtype))
                self.ln_out = LayerNorm(
                    w["ln_out.weight"],
                    w["ln_out.bias"]
                )
                self.ln_in = LayerNorm(
                    w["blocks.0.ln0.weight"],
                    w["blocks.0.ln0.bias"]
                )

                self.head = MM8(
                    w["head.weight"].to(dtype), runtimedtype)
                # loading bar
                from tqdm import tqdm
                self.blocks = torch.nn.ModuleList([Block(dims,w, i) for i in tqdm(range(layers), desc="loading layers")])

               

                self.device = device

                # del w
                # torch.cuda.empty_cache()
                # gc.collect()
            @ method
            def forward(self, x, state):
                
                x = self.emb(x)
                x = self.ln_in(x)

                for i, block in enumerate(self.blocks):

                    x, rstate = block(x, state[i])
                    state[i] = rstate

                x = self.ln_out(x)

                outx = self.head(x[-1:]).detach().squeeze()
                

                return outx, state
            
        if path.endswith(".pt"):
            myrwkv = torch.jit.load(path)
            returnObject: myRWKV = myrwkv
            return returnObject
        
        myrwkv = myRWKV(path)
        
        # print the memory used in gb
        print("Memory used: ", torch.cuda.memory_allocated()/1024**3, "GB")

        myrwkv = torch.jit.script(myrwkv)
        print("Memory used: ", torch.cuda.memory_allocated()/1024**3, "GB")

        # torch.jit.save(myrwkv, "myrwkv.pt")
        returnObject: myRWKV = myrwkv

        return returnObject
