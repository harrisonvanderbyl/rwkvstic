
from rwkvstic.agnostic.backends.modules.base import RwkvModule
import torch
class MM8(RwkvModule):
            def __init__(self, weight, device, maxVram):
                
                super(MM8, self).__init__()
                self.runtimedtype = torch.float32
                

                

                
                
                self.weight, self.range, self.offset = self.chunkQuantizeMatrix(weight,device=device, maxvram=maxVram)
                
                
                self.device = device
                self.range = self.range.to(device)
                self.offset = self.offset.to(device)
                
                self.M = self.weight.shape[0]
                self.N = self.weight.shape[1]

                del weight
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            def chunkQuantizeMatrix(self, x, splits=32, device="cuda", maxvram=100):
                todev = "cpu"
                if torch.cuda.memory_allocated(device) / 1024 ** 3 < maxvram:
                         print((torch.cuda.memory_allocated(device) / 1024**3) , "GB allocated")
                         todev = device
                toset = torch.empty(x.shape[::-1], device=todev, dtype=torch.uint8)
                splitmatrices = range(splits)
                xx = [self.QuantizeMatrix(x.t(), m, toset)
                    for m in splitmatrices]
                mrange = (torch.cat([x[0] for x in xx])).float()
                offset = (torch.cat([x[1] for x in xx])).float()

                return toset, mrange, offset
            def QuantizeMatrix(self, xx, i, toset):
                width = xx.shape[0]
                start = i * width
                end = start + width
                x = xx[start:end].t()
                rang = 255
                ran, mini = (x.max(0)[0]-x.min(0)[0])/rang,  x.min(0)[0]
                toset[start:end] = ((x-mini)/ran).round().t().to(torch.uint8).to(toset.device, non_blocking=True)
                return [ran.to(self.runtimedtype).clone(), mini.to(self.runtimedtype).clone()]

            @ torch.jit.script_method
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

            @ torch.jit.script_method
            def forward(self, y):
                
                B = y.shape[0]
                yy = y.to(self.runtimedtype)
                if B > 1:
                    xmain = (yy*self.range).to(torch.float16) @ self.weight.to(dtype=torch.float16,device=self.device)
                    xmain = xmain.to(self.runtimedtype)
                    zp = (y.mv(self.offset)).reshape(-1, 1)
                    
                     
                    return xmain + zp 
                zp = (y.mul(self.offset).sum()).to(self.runtimedtype)
                xmain = self.cuda_mm8(self.N, self.M, yy, self.weight.to(self.device),self.range).to(self.runtimedtype)
                

                #
                return (xmain + zp)
            

            
                
class MM8_3(MM8):
            def __init__(self, weight,weight1,weight2, device, maxVram):
                
                super(MM8_3, self).__init__(
                     weight,
                        device,
                        maxVram
                )

                self.weight1, self.range1, self.offset1 = self.chunkQuantizeMatrix(weight1,device=device, maxvram=maxVram)
                
                
                self.weight2, self.range2, self.offset2 = self.chunkQuantizeMatrix(weight2,device=device, maxvram=maxVram)
                del weight
                del weight1
                del weight2
                self.range1 = self.range1.to(device)
                self.offset1 = self.offset1.to(device)
                self.range2 = self.range2.to(device)
                self.offset2 = self.offset2.to(device)
                
            
            
                            
            @ torch.jit.script_method
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

            @ torch.jit.script_method
            def forward(self, y):
                
                B = y.shape[1]
                yy = y.to(self.runtimedtype)
                if B > 1:
                    xmain = ((yy[0]*self.range).to(torch.float16) @ self.weight.to(self.device).to(torch.float16)).to(self.runtimedtype)
                    xmain1 = ((yy[1]*self.range1).to(torch.float16) @ self.weight1.to(self.device).to(torch.float16)).to(self.runtimedtype)
                    xmain2 = ((yy[2]*self.range2).to(torch.float16) @ self.weight2.to(self.device).to(torch.float16)).to(self.runtimedtype)
 
                    zp = (y[0].mv(self.offset)).reshape(-1, 1)
                    zp1 = (y[1].mv(self.offset1)).reshape(-1, 1)
                    zp2 = (y[2].mv(self.offset2)).reshape(-1, 1)
                    
                     
                    return xmain + zp , xmain1 + zp1, xmain2 + zp2
                zp = (y[0].mul(self.offset).sum()).to(self.runtimedtype)
                zp1 = (y[1].mul(self.offset1).sum()).to(self.runtimedtype)
                zp2 = (y[2].mul(self.offset2).sum()).to(self.runtimedtype)               
                xmain = self.cuda_mm83(self.N, self.M, yy, self.weight.to(self.device), self.weight1.to(self.device), self.weight2.to(self.device),self.range,self.range1,self.range2)
                


                return (xmain[0] + zp), (xmain[1] + zp1), (xmain[2] + zp2)
        