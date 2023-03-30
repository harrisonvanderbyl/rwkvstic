
from rwkvstic.agnostic.backends.modules.base import RwkvModule
import torch
class MM8(RwkvModule):
            def __init__(self, weight, device):
                super(MM8, self).__init__()
                self.device = device
                self.weight, self.range, self.offset = self.chunkQuantizeMatrix(weight)
 
                self.range = self.range.to(device)
                self.offset = self.offset.to(device)
                self.M = self.weight.shape[0]
                self.N = self.weight.shape[1]

            def chunkQuantizeMatrix(self, x):
                toset = torch.empty(x.shape[::-1], device=self.device, dtype=torch.uint8)
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
                return [ran.to(torch.float32).to(self.device).clone(), mini.to(torch.float64).to(self.device).clone()]

            def cuda_mm8(self, N:int, M:int, x, w, r):
                x = x[0]
                y = torch.zeros(N, device=self.device, dtype=torch.float32)
                torch.ops.rwkv.mm8_one(M,N, x, w, y,self.range)
               
                y = y.to(torch.float64)

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
            def __init__(self, weight,weight1,weight2, device):   
                super(MM8_3, self).__init__(weight,device)
                self.weight1, self.range1, self.offset1 = self.chunkQuantizeMatrix(weight1) 
                self.weight2, self.range2, self.offset2 = self.chunkQuantizeMatrix(weight2)
                
            def cuda_mm83(self, x):
            
                x0 = x[0].squeeze().contiguous().to(torch.float32)
                x1 = x[1].squeeze().contiguous().to(torch.float32)
                x2 = x[2].squeeze().contiguous().to(torch.float32)
                y = torch.zeros(self.N, device=self.device, dtype=torch.float32)
                y1 = torch.zeros(self.N, device=self.device, dtype=torch.float32)
                y2 = torch.zeros(self.N, device=self.device, dtype=torch.float32)
                torch.ops.rwkv.mm8_three(self.M,self.N, x0,x1,x2, self.weight,self.weight1,self.weight2, y,y1,y2,self.range,self.range1,self.range2)
                
                y = y.to(dtype=torch.float64)
                y1 = y1.to(dtype=torch.float64)
                y2 = y2.to(dtype=torch.float64)

                return y.unsqueeze(0),y1.unsqueeze(0),y2.unsqueeze(0)

            def forward(self, y):
                
                B = y.shape[1]
                y = y.to(torch.float64)
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
        
class Linear(RwkvModule):
        def __init__(self,w):
            super().__init__()
            self.w = w.t().clone()
            self.device = self.w.device
            
        def func(self,x): return x.to(device=self.w.device, dtype=self.w.dtype) @ self.w

        def forward(self,x):
            return self.func(x).to(torch.float64)

        def config (self, devices, **kwargs):
            splits = devices.__len__()
            weights = torch.chunk(self.w.t(), splits, dim=1)
            funcs = []
            for i, weight in enumerate(weights):
                device = devices[i]["device"]
                dtype = devices[i]["dtype"]
                if dtype == torch.uint8:
                    funcs.append(MM8(weight, device))
                else:
                    funcs.append(Linear(weight.to(device=device, dtype=dtype)))
                    
            self.weights = funcs 
            def newfunc(x):
                splits = torch.chunk(x, self.weights.__len__(), dim=1)
                outs = 0
                for i, split in enumerate(splits):
                    outs += self.weights[i](split.to(self.weights[i].device)).to(dtype=torch.float64, device=devices[0]["device"])

                return outs
            self.func = newfunc

class Linear3(RwkvModule):
        def __init__(self,w,w1,w2):
            super().__init__()
            self.w = w.t().clone()
            self.w1 = w1.t().clone()
            self.w2 = w2.t().clone()
            self.device = self.w.device
            # self.w = Linear(w)
            # self.w1 = Linear(w1)
            # self.w2 = Linear(w2)
        def func(self,x): 
            # return self.w(x[0]), self.w1(x[1]), self.w2(x[2])
            return x[0].to(device=self.w.device, dtype=self.w.dtype) @ self.w , x[1].to(device=self.w1.device, dtype=self.w1.dtype) @ self.w1 , x[2].to(device=self.w2.device, dtype=self.w2.dtype) @ self.w2
            # return x[0].to(device=self.w.device, dtype=self.w.dtype)@self.w , x[1].to(device=self.w1.device, dtype=self.w1.dtype)@self.w1 , x[2].to(device=self.w2.device, dtype=self.w2.dtype)@self.w2
            
        def forward(self,x):
            a,b,c = self.func(x)
            return a.to(torch.float64),b.to(torch.float64),c.to(torch.float64)

        def config (self,devices, **config):
            # self.w.config(**config)
            # self.w1.config(**config)
            # self.w2.config(**config)

            splits = devices.__len__()
            weights = torch.chunk(self.w.t(), splits, dim=1)
            weights1 = torch.chunk(self.w1.t(), splits, dim=1)
            weights2 = torch.chunk(self.w2.t(), splits, dim=1)
            funcs = []
            for i, weight in enumerate(weights):
                device = devices[i]["device"]
                dtype = devices[i]["dtype"]
                if dtype == torch.uint8:
                    funcs.append(MM8_3(weight, weights1[i], weights2[i], device))
                else:
                    funcs.append(Linear3(weight.to(device=device, dtype=dtype), weights1[i].to(device=device, dtype=dtype), weights2[i].to(device=device, dtype=dtype)))
                    
            self.weights = funcs 
            def newfunc(x):
                splits = torch.chunk(x[0], self.weights.__len__(), dim=1)
                splits1 = torch.chunk(x[1], self.weights.__len__(), dim=1)
                splits2 = torch.chunk(x[2], self.weights.__len__(), dim=1)
                outs = 0
                outs1 = 0
                outs2 = 0
                for i, split in enumerate(splits):
                    outs0, outs01, outs02 = self.weights[i](torch.stack([split.to(self.weights[i].device), splits1[i].to(self.weights[i].device), splits2[i].to(self.weights[i].device)]))
                    outs += outs0.to(dtype=torch.float64, device=devices[0]["device"])
                    outs1 += outs01.to(dtype=torch.float64, device=devices[0]["device"])
                    outs2 += outs02.to(dtype=torch.float64, device=devices[0]["device"])

                return outs, outs1, outs2
            self.func = newfunc
