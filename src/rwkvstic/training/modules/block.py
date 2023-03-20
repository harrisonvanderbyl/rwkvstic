
from rwkvstic.training.modules.wkv import wkv_power
import torch
class Block(torch.nn.Module):
            def __init__(self, dims,T):
                super(Block, self).__init__()

                
                self.ln1 = torch.nn.LayerNorm((dims,))
                self.ln2 = torch.nn.LayerNorm((dims,))
               
                
                self.attkey = torch.nn.Linear(dims, dims)
                self.attvalue = torch.nn.Linear(dims, dims)
                self.attreceptance = torch.nn.Linear(dims, dims)

                self.ffnkey = torch.nn.Linear(dims, dims)
                self.ffnvalue = torch.nn.Linear(dims, dims)
                self.attout = torch.nn.Linear(dims, dims)
                self.ffnreceptance = torch.nn.Linear(dims, dims)
                
                self.attmixk = torch.nn.Parameter(torch.randn(dims))
                self.attmixv = torch.nn.Parameter(torch.randn(dims))
                self.attmixr = torch.nn.Parameter(torch.randn(dims))
                
                self.wkv = wkv_power(dims,T)

                # self.t = [powerTri(self.time_decay, i) for i in range(1, 21)]


                self.ffnmixk = torch.nn.Parameter(torch.randn(dims))
                self.ffnmixr = torch.nn.Parameter(torch.randn(dims))

                # set all weights to 0
                self.attkey.weight.data.zero_()
                self.attvalue.weight.data.zero_()
                self.attreceptance.weight.data.zero_()
                self.ffnkey.weight.data.zero_()
                self.ffnvalue.weight.data.zero_()
                self.attout.weight.data.zero_()
                self.ffnreceptance.weight.data.zero_()
                self.attmixk.data.zero_()
                self.attmixv.data.zero_()
                self.attmixr.data.zero_()
                self.ffnmixk.data.zero_()
                self.ffnmixr.data.zero_()
                self.wkv.time_first.data.zero_()
                self.wkv.time_decay.data.zero_()

                self.register_module("ln1", self.ln1)
                self.register_module("ln2", self.ln2)
                self.register_module("attkey", self.attkey)
                self.register_module("attvalue", self.attvalue)
                self.register_module("attreceptance", self.attreceptance)
                self.register_module("ffnkey", self.ffnkey)
                self.register_module("ffnvalue", self.ffnvalue)
                self.register_module("attout", self.attout)
                self.register_module("ffnreceptance", self.ffnreceptance)
                self.register_module("wkv", self.wkv)
                
                self.register_parameter("attmixk", self.attmixk)
                self.register_parameter("attmixv", self.attmixv)
                self.register_parameter("attmixr", self.attmixr)
                self.register_parameter("ffnmixk", self.ffnmixk)
                self.register_parameter("ffnmixr", self.ffnmixr)

                torch.cuda.empty_cache()
            
            
            def forward(self, x):

                xy = self.ln1(x)

                tc = xy.roll(1, 0)
                # rmc = tc[0].clone() for inference
                # tc[0] = state[0]
                # state[0] = rmc

                mixk = torch.lerp(tc, xy, self.attmixk)
                mixv = torch.lerp(tc, xy, self.attmixv)
                mixr = torch.lerp(tc, xy, self.attmixr)

                k,v,r = self.attkey(mixk), self.attvalue(mixv), self.attreceptance(mixr)


                r = r.sigmoid()

                # WKV kernel

                wkv = self.wkv(k, v)
               
                wkv *= r

                rz = x + self.attout(wkv)

                ddd = self.ln2(rz)

                rc = ddd.roll(1, 0)
                # dc = rc[0].clone()
                # rc[0] = state[1]
                # state[1] = dc for inference

                fmixk = torch.lerp(rc, ddd, self.ffnmixk)
                fmixr = torch.lerp(rc, ddd, self.ffnmixr)

                

                rf = self.ffnreceptance(fmixr).sigmoid()
                
                kf = self.ffnkey(fmixk).relu()
                rvm = self.ffnvalue(torch.square(kf))

                out = rvm * rf + rz

                return out

                # stuff
