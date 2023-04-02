
from rwkvstic.agnostic.backends.modules.base import RwkvModule
import torch
from typing import List
class MM8(RwkvModule):
            def __init__(self, weight, device, stream = False):
                super(MM8, self).__init__()
                self.device = device
                self.stream = stream
                self.runtimedtype = torch.float32 if self.device == "mps" else torch.float64
                
                self.weight, self.range, self.offset = self.chunkQuantizeMatrix(weight)
                self.range = self.range.to(device)
                self.offset = self.offset.to(device)
                self.M = self.weight.shape[0]
                self.N = self.weight.shape[1]

            def chunkQuantizeMatrix(self, x):
                toset = torch.empty(x.shape[::-1], device=self.device if not self.stream else "cpu", dtype=torch.uint8)
                xx = self.QuantizeMatrix(x.t(), 0, toset)
                mrange = xx[0]
                offset = xx[1]
                return toset, mrange, offset
            
            def QuantizeMatrix(self, xx, i, toset):
                width = xx.shape[0]
                start = i * width
                end = start + width
                x = xx[start:end].t()
                rang = 255
                ran, mini = (x.max(0)[0]-x.min(0)[0])/rang,  x.min(0)[0]
                toset[start:end] = ((x-mini)/ran).round().t().to(torch.uint8).to(toset.device, non_blocking=True)
                return [ran.to(torch.float32).to(self.device).clone(), mini.to(self.runtimedtype).to(self.device).clone()]

            def cuda_mm8(self, N:int, M:int, x, w, r):
                x = x[0]
                y = torch.zeros(N, device=self.device, dtype=torch.float32)
                torch.ops.rwkv.mm8_one(M,N, x, w, y,self.range)
               
            

                return y.unsqueeze(0)

            def forward(self, y):
                
                B = y.shape[0]
                y = y.to(dtype=torch.float32,device=self.device)
               
                if B > 1:
                    return ((y*self.range).to(dtype=torch.bfloat16) @ self.weight.to(dtype=torch.bfloat16)).to(dtype=torch.float64) + (y.mv(self.offset)).reshape(-1, 1)
                
                zp = y.mul(self.offset).sum()
                xmain = self.cuda_mm8(self.N, self.M, y, self.weight,self.range)

                return (xmain + zp)
            

            
                
class MM8_3(MM8):
            def __init__(self, weight,weight1,weight2, device, stream = False):   
                super(MM8_3, self).__init__(weight,device, stream)
                self.weight1, self.range1, self.offset1 = self.chunkQuantizeMatrix(weight1) 
                self.weight2, self.range2, self.offset2 = self.chunkQuantizeMatrix(weight2)
                
            def cuda_mm83(self, x:List[torch.Tensor]):
            
                x0 = x[0].squeeze().contiguous().to(torch.float32)
                x1 = x[1].squeeze().contiguous().to(torch.float32)
                x2 = x[2].squeeze().contiguous().to(torch.float32)
                y = torch.zeros(self.N, device=self.device, dtype=torch.float32)
                y1 = torch.zeros(self.N, device=self.device, dtype=torch.float32)
                y2 = torch.zeros(self.N, device=self.device, dtype=torch.float32)
                torch.ops.rwkv.mm8_three(self.M,self.N, x0,x1,x2, self.weight,self.weight1,self.weight2, y,y1,y2,self.range,self.range1,self.range2)

                return y.unsqueeze(0),y1.unsqueeze(0),y2.unsqueeze(0)

            def forward(self, y:List[torch.Tensor]):
                
                B = y[0].shape[0]
                
                if B > 1:
                    xmain = ((y[0]*self.range.to(torch.float64)).to(torch.bfloat16) @ self.weight.to(self.device).to(torch.bfloat16)).to(torch.float64)
                    xmain1 = ((y[1]*self.range1.to(torch.float64)).to(torch.bfloat16) @ self.weight1.to(self.device).to(torch.bfloat16)).to(torch.float64)
                    xmain2 = ((y[2]*self.range2.to(torch.float64)).to(torch.bfloat16) @ self.weight2.to(self.device).to(torch.bfloat16)).to(torch.float64)
 
                    zp = (y[0].mv(self.offset)).reshape(-1, 1)
                    zp1 = (y[1].mv(self.offset1)).reshape(-1, 1)
                    zp2 = (y[2].mv(self.offset2)).reshape(-1, 1)
                    return xmain + zp , xmain1 + zp1, xmain2 + zp2
                zp = (y[0].mul(self.offset).sum())
                zp1 = (y[1].mul(self.offset1).sum())
                zp2 = (y[2].mul(self.offset2).sum())             
                xmain = self.cuda_mm83(y)
                


                return (xmain[0] + zp), (xmain[1] + zp1), (xmain[2] + zp2)

class Linear3Naive(RwkvModule):
        def __init__(self,w,w1,w2):
            super().__init__()
            self.w = w.t()
            self.w1 = w1.t()
            self.w2 = w2.t()
            self.device = self.w.device
        
        def forward(self,x:List[torch.Tensor]):
            return x[0].to(device=self.w.device, dtype=self.w.dtype) @ self.w , x[1].to(device=self.w1.device, dtype=self.w1.dtype) @ self.w1 , x[2].to(device=self.w2.device, dtype=self.w2.dtype) @ self.w2
   

class LinearNaive(RwkvModule):
        def __init__(self,w):
            super().__init__()
            self.w = w.t()
            self.device = self.w.device
        
        def forward(self,x):
            return x.to(device=self.w.device, dtype=self.w.dtype) @ self.w
        
class mm8Naive(MM8):
        def __init__(self,w, device, dtype, stream):
            self.dtype = dtype
            self.device = device
            super().__init__(w, device, stream)
        
        def forward(self,x):
            return ((x*self.range).to(device=self.device, dtype=self.dtype) @ self.weight.to(device=self.device,dtype=self.dtype)) + (x.mv(self.offset)).reshape(-1, 1)
 
class mm8_3Naive(MM8_3):
        def __init__(self,w,w1,w2, device, dtype, stream):
            self.dtype = dtype
            self.device = device
            super().__init__(w,w1,w2, device, stream)
        
        def forward(self,x):
            return ((x[0]*self.range).to(device=self.device, dtype=self.dtype) @ self.weight.to(device=self.device,dtype=self.dtype)) + (x[0].mv(self.offset)).reshape(-1, 1), \
                    (x[1]*self.range1).to(device=self.device, dtype=self.dtype) @ self.weight1.to(device=self.device,dtype=self.dtype) + (x[1].mv(self.offset1)).reshape(-1, 1), \
                    (x[2]*self.range2).to(device=self.device, dtype=self.dtype) @ self.weight2.to(device=self.device,dtype=self.dtype) + (x[2].mv(self.offset2)).reshape(-1, 1)

class Linear(RwkvModule):
        def __init__(self,w:torch.Tensor):
            super().__init__()
            self.w = w.t()
            self.weights: torch.jit.RecursiveScriptModule = torch.nn.ModuleList()
            
            self.device = self.w.device
            self.runtimedtype = torch.float32 if self.device == "mps" else torch.float64
        

        def newfunc(self, x):
            splits = torch.chunk(x, self.weights.__len__(), dim=1)
            
            splits = [splits[i].to(weight.device,dtype=self.runtimedtype, non_blocking=True) for i,weight in enumerate(self.weights)]
            
            outlist = [weight(splits[i]) for i,weight in enumerate(self.weights)]
            out = torch.zeros_like(outlist[0])
            for i, outo in enumerate(outlist):
                out += outo.to(dtype=torch.float64, device=self.device, non_blocking=True)
            return out
        
        
        def forward(self,x):
            return self.newfunc(x).to(self.runtimedtype)

        
        def delw(self):
            self.w = torch.tensor(0)

        
        def fillSubWeights(self, weights:torch.nn.modules.container.ModuleList):
            self.weights = weights
        
        def config (self, devices, **kwargs):
            splits = devices.__len__()
            weights = torch.chunk(self.w.t().clone(), splits, dim=1)
            funcs = []
            for i, weight in enumerate(weights):
                device = devices[i]["device"]
                naive = devices[i].get("naive", device == "cpu" or device == "mps")
                naivedtype = devices[i].get("naivedtype", torch.float32 if device == "cpu" else torch.float16)
                dtype = devices[i].get("dtype", torch.uint8)
                stream = devices[i].get("stream", False)
                if dtype == torch.uint8:
                    if naive:
                        funcs.append(mm8Naive(weight,device, naivedtype, stream))
                    else:
                        funcs.append(MM8(weight, device,stream))
                else:
                    funcs.append(LinearNaive(weight.to(device=device, dtype=dtype)))
            XX = self.w.shape[1]
            self.delw()
            torch.cuda.empty_cache()
            
            self.fillSubWeights(torch.nn.ModuleList(funcs))
            self.device = devices[0]["device"]
            if torch.device(self.device).type == "mps":
                self.runtimedtype = torch.float32

class Linear3(RwkvModule):
        def __init__(self,w,w1,w2):
            super().__init__()
            self.w = w.t()
            self.w1 = w1.t()
            self.w2 = w2.t()
            self.device = self.w.device
            self.runtimedtype = torch.float64
            self.weights = torch.nn.ModuleList()
        
        
        def newfunc(self,x:torch.Tensor):
            splits = torch.chunk(x[0], self.weights.__len__(), dim=1)
            splits1 = torch.chunk(x[1], self.weights.__len__(), dim=1)
            splits2 = torch.chunk(x[2], self.weights.__len__(), dim=1)
            
            splits = [splits[i].to(weight.device,dtype=self.runtimedtype, non_blocking=True) for i,weight in enumerate(self.weights)]
            splits1 = [splits1[i].to(weight.device,dtype=self.runtimedtype, non_blocking=True) for i,weight in enumerate(self.weights)]
            splits2 = [splits2[i].to(weight.device,dtype=self.runtimedtype, non_blocking=True) for i,weight in enumerate(self.weights)]
            
            outlist = [weight(([splits[i], splits1[i], splits2[i]])) for i,weight in enumerate(self.weights)]
         
            out = torch.zeros_like(outlist[0][0])
            out1 = torch.zeros_like(outlist[0][1])
            out2 = torch.zeros_like(outlist[0][2])
            for i, outo in enumerate(outlist):
            
                out += outo[0].to(dtype=self.runtimedtype, device=self.device, non_blocking=True)
                out1 += outo[1].to(dtype=self.runtimedtype, device=self.device, non_blocking=True)
                out2 += outo[2].to(dtype=self.runtimedtype, device=self.device, non_blocking=True)

            

            return out, out1, out2
    
            
        def forward(self,x:torch.Tensor):
            a,b,c = self.newfunc(x)
            return a.to(self.runtimedtype),b.to(self.runtimedtype),c.to(self.runtimedtype)
        
        
        
        def delw(self):
            self.w = torch.tensor(0)
            self.w1 = torch.tensor(0)
            self.w2 = torch.tensor(0)

        
        def fillSubWeights(self, weights:torch.nn.ModuleList):
           
            self.weights = weights

            len(self.weights)

        def config (self,devices, **config):

            splits = devices.__len__()
            weights = torch.chunk(self.w.t(), splits, dim=1)
            weights1 = torch.chunk(self.w1.t(), splits, dim=1)
            weights2 = torch.chunk(self.w2.t(), splits, dim=1)
            funcs = torch.nn.ModuleList()
            for i, weight in enumerate(weights):
                device = devices[i]["device"]
                naive = devices[i].get("naive", device == "cpu" or device == "mps")
                dtype = devices[i]["dtype"]
                naivedtype = devices[i].get("naivedtype", torch.float32 if device == "cpu" else torch.float16)
                stream = devices[i].get("stream", False)
                if dtype == torch.uint8:
                    if naive:
                        funcs.append(mm8_3Naive(weight, weights1[i], weights2[i], device, naivedtype, stream))
                    else: 
                        funcs.append(MM8_3(weight, weights1[i], weights2[i], device,stream))
                else:
                    funcs.append(Linear3Naive(weight.to(device=device, dtype=dtype), weights1[i].to(device=device, dtype=dtype), weights2[i].to(device=device, dtype=dtype)))
            self.delw()
            torch.cuda.empty_cache()
            self.fillSubWeights(funcs)
            self.device = devices[0]["device"]
            if torch.device(self.device).type == "mps":
                self.runtimedtype = torch.float32
