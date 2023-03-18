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

method = torch.jit.script_method
module = torch.jit.ScriptModule
script = torch.jit.script
with torch.no_grad():
    def OptRWKV(path, jit=True, export=False, **kwargs):
      

        device = "cuda"

        dtype = torch.float32
        runtimedtype = torch.float32

        load(
            name=f"wkv_cuda",
            sources=[f"{current_path}/cuda/wrapper.cpp",
                    f"{current_path}/cuda/operators.cu"],
            verbose=False,
            extra_cuda_cflags=["-std=c++17", "-O3" ] + (["-gencode", "arch=compute_75,code=sm_75"] if torch.cuda.get_device_capability(0)[0] >= 7 else []) + (["-gencode", "arch=compute_80,code=sm_80"] if torch.cuda.get_device_capability(0)[0] >= 8 else []) + (["-gencode", "arch=compute_86,code=sm_86"] if torch.cuda.get_device_capability(0)[0] >= 8 else []),
            
            is_python_module=False)
        
        
 
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
                return [x, ran.to(self.runtimedtype), mini.to(self.runtimedtype)]
            @ method
            def cuda_mm8(self, N:int, M:int, x, w, r):
                # assert B == 1
                # assert w.dtype == torch.uint8
                x = x[0]
                # assert x.shape == [M]
                
                # assert w.shape == [M, N]
                y = torch.zeros(N, device=w.device, dtype=self.runtimedtype)
                torch.ops.rwkv.mm8_one(M,N, x, w, y,r)
                
                y = y.to(dtype=self.runtimedtype)

                return y.unsqueeze(0)

            @ method
            def forward(self, y):
                
                B = y.shape[0]
                yy = y.to(self.runtimedtype)
                if B > 1:
                    xmain = (yy*self.range).to(torch.float16) @ self.weight.to(torch.float16)
                    xmain = xmain.to(self.runtimedtype)
                    zp = (y.mv(self.offset)).reshape(-1, 1)
                    
                     
                    return xmain + zp 
                zp = (y.mul(self.offset).sum()).to(self.runtimedtype)
                xmain = self.cuda_mm8(self.N, self.M, yy, self.weight,self.range).to(self.runtimedtype)
                

                #
                return (xmain + zp)
        class MM8_3(module):
            def __init__(self, weight,weight1,weight2, runtimedtype):
                
                super(MM8_3, self).__init__()

                self.runtimedtype = runtimedtype
                splitmatrices = torch.chunk(weight, 32, 1)

                
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
                
                splitmatrices1 = torch.chunk(weight1, 32, 1)
                xx1 = [self.QuantizeMatrix(x)
                    for x in splitmatrices1]
                self.weight1 = (torch.cat([x[0] for x in xx1], 1).t().contiguous()).to (
                    device=device)
                self.range1 = (torch.cat([x[1] for x in xx1])).to (
                    device=device)
                self.offset1 = (torch.cat([x[2] for x in xx1])).to (
                    device=device)
                
                splitmatrices2 = torch.chunk(weight2, 32, 1)
                xx2 = [self.QuantizeMatrix(x)
                    for x in splitmatrices2]
                self.weight2 = (torch.cat([x[0] for x in xx2], 1).t().contiguous()).to (
                    device=device)
                self.range2 = (torch.cat([x[1] for x in xx2])).to (
                    device=device)
                self.offset2 = (torch.cat([x[2] for x in xx2])).to (
                    device=device)
                
            
            def QuantizeMatrix(self, x):
                rang = 255
                ran, mini = (x.max(0)[0]-x.min(0)[0])/rang,  x.min(0)[0]
                x = x.double()
                x = ((x-mini)/ran).round()
                x = x.to(
                        dtype=torch.uint8)
                return [x, ran.to(self.runtimedtype), mini.to(self.runtimedtype)]
            @ method
            def cuda_mm83(self, N:int, M:int, x, w,w1,w2, r,r1,r2):
            
                x0 = x[0].squeeze().contiguous()
                x1 = x[1].squeeze().contiguous()
                x2 = x[2].squeeze().contiguous()
                y = torch.zeros(N, device=w.device, dtype=self.runtimedtype)
                y1 = torch.zeros(N, device=w.device, dtype=self.runtimedtype)
                y2 = torch.zeros(N, device=w.device, dtype=self.runtimedtype)
                torch.ops.rwkv.mm8_three(M,N, x0,x1,x2, w,w1,w2, y,y1,y2,r,r1,r2)
                
                y = y.to(dtype=self.runtimedtype)
                y1 = y1.to(dtype=self.runtimedtype)
                y2 = y2.to(dtype=self.runtimedtype)

                return y.unsqueeze(0),y1.unsqueeze(0),y2.unsqueeze(0)

            @ method
            def forward(self, y):
                
                B = y.shape[1]
                yy = y.to(self.runtimedtype)
                if B > 1:
                    xmain = ((yy[0]*self.range).to(torch.float16) @ self.weight.to(torch.float16)).to(self.runtimedtype)
                    xmain1 = ((yy[1]*self.range1).to(torch.float16) @ self.weight1.to(torch.float16)).to(self.runtimedtype)
                    xmain2 = ((yy[2]*self.range2).to(torch.float16) @ self.weight2.to(torch.float16)).to(self.runtimedtype)
 
                    zp = (y[0].mv(self.offset)).reshape(-1, 1)
                    zp1 = (y[1].mv(self.offset1)).reshape(-1, 1)
                    zp2 = (y[2].mv(self.offset2)).reshape(-1, 1)
                    
                     
                    return xmain + zp , xmain1 + zp1, xmain2 + zp2
                zp = (y[0].mul(self.offset).sum()).to(self.runtimedtype)
                zp1 = (y[1].mul(self.offset1).sum()).to(self.runtimedtype)
                zp2 = (y[2].mul(self.offset2).sum()).to(self.runtimedtype)               
                xmain = self.cuda_mm83(self.N, self.M, yy, self.weight, self.weight1, self.weight2,self.range,self.range1,self.range2)
                


                return (xmain[0] + zp), (xmain[1] + zp1), (xmain[2] + zp2)
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
               
                
                self.att = MM8_3(
                    w[f"blocks.{i}.att.key.weight"],
                    w[f"blocks.{i}.att.value.weight"],
                    w[f"blocks.{i}.att.receptance.weight"], runtimedtype)

                self.ffnkey = MM8(
                    w[f"blocks.{i}.ffn.key.weight"], runtimedtype)
                self.ffnvalue = MM8(
                    w[f"blocks.{i}.ffn.value.weight"], runtimedtype)
                self.attout = MM8(
                    w[f"blocks.{i}.att.output.weight"], runtimedtype)
                self.ffnreceptance = MM8(
                    w[f"blocks.{i}.ffn.receptance.weight"], runtimedtype)
                
                self.attmix= torch.stack((w[f"blocks.{i}.att.time_mix_k"].squeeze(),
                  w[f"blocks.{i}.att.time_mix_v"].squeeze(),
                     w[f"blocks.{i}.att.time_mix_r"].squeeze())).to(runtimedtype).to (
                    device=device).unsqueeze(1)
                
                self.time_first = w[f"blocks.{i}.att.time_first"].squeeze().to(runtimedtype).to (
                    device=device)

                self.time_decay = w[f"blocks.{i}.att.time_decay"].squeeze().to(runtimedtype).to (
                    device=device).double().exp().neg()

                # self.t = [powerTri(self.time_decay, i) for i in range(1, 21)]


                self.ffnmix= torch.stack((w[f"blocks.{i}.ffn.time_mix_k"].squeeze(),
                    w[f"blocks.{i}.ffn.time_mix_r"].squeeze())).to(runtimedtype).to (
                    device=device).unsqueeze(1)
                torch.cuda.empty_cache()
                print("Memory allocated after layer: "+str(i), torch.cuda.memory_allocated(device=device)/1024/1024, "MB")
            @method
            def cuda_wkv(self, T: int, C: int, w, u, k, v, aa, bb, pp):
                assert 1 * C % min(C, 32) == 0
                assert k.dtype == self.runtimedtype
                w = w.contiguous()
                u = u.contiguous()
                k = k.contiguous()
                v = v.contiguous()
                y = torch.empty((T, C), device="cuda", memory_format=torch.contiguous_format, dtype=self.runtimedtype)
            
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

                k,v,r = self.att(mix)


                r = r.sigmoid()

                # WKV kernel original

                wkv, state[2],state[3],state[4] = self.cuda_wkv(k.shape[0], k.shape[1], self.time_decay.float(), self.time_first.float(), k, v, state[2].float(), state[3].float(), state[4].float())
               
                wkv *= r

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
                
                # self.device = device
                self.stordev = "cpu"
                self.w = w.to (
                    device=self.stordev)
                self.rundev = device
                self.runtimedtype = runtimedtype
            def forward(self,x):
                return self.w[x.to(
                    device=self.stordev)].to(self.runtimedtype).to(
                    device=self.rundev)
            
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

                

                
               
                self.emb =  RwkvEmb(w["emb.weight"])
                self.ln_out = LayerNorm(
                    w["ln_out.weight"],
                    w["ln_out.bias"]
                )
                self.ln_in = LayerNorm(
                    w["blocks.0.ln0.weight"],
                    w["blocks.0.ln0.bias"]
                )

                self.head = MM8(
                    w["head.weight"], runtimedtype)
                
                print("Memory allocated before layers: ", torch.cuda.memory_allocated(device=device)/1024/1024, "MB")
                # loading bar
                from tqdm import tqdm
                self.blocks = torch.nn.ModuleList([Block(dims,w, i) for i in tqdm(range(layers), desc="loading layers")])

               

                self.device = device

                del w
                torch.cuda.empty_cache()
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
            
        if path.endswith(".rwkv"):
            myrwkv = torch.jit.load(path)
            returnObject: myRWKV = myrwkv
            return returnObject
        
        myrwkv = myRWKV(path)
        
        if jit:
            myrwkv = torch.jit.script(myrwkv)
        if export:
            torch.jit.save(myrwkv, export+".rwkv")
            
            exit()
        returnObject: myRWKV = myrwkv
        return returnObject
