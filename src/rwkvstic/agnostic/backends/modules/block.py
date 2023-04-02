from rwkvstic.agnostic.backends.modules.base import RwkvModule
from rwkvstic.agnostic.backends.modules.layernorm import LayerNorm
from rwkvstic.agnostic.backends.modules.matmul import Linear, Linear3
from rwkvstic.agnostic.backends.modules.wkv import WKV
import torch
class Block(RwkvModule):
            def __init__(self,w, i):
                super(Block, self).__init__()

                self.t = []

                
                self.ln1 = LayerNorm(
                    w[f"blocks.{i}.ln1.weight"], w[f"blocks.{i}.ln1.bias"])
                self.ln2 = LayerNorm(
                    w[f"blocks.{i}.ln2.weight"], w[f"blocks.{i}.ln2.bias"])
               
                
                self.att = Linear3(
                    w[f"blocks.{i}.att.key.weight"],
                    w[f"blocks.{i}.att.value.weight"],
                    w[f"blocks.{i}.att.receptance.weight"])

                self.ffnkey = Linear(
                    w[f"blocks.{i}.ffn.key.weight"])
                self.ffnvalue = Linear(
                    w[f"blocks.{i}.ffn.value.weight"])
                self.attout = Linear(
                    w[f"blocks.{i}.att.output.weight"])
                self.ffnreceptance = Linear(
                    w[f"blocks.{i}.ffn.receptance.weight"])
                
                self.wkv = WKV()
                
                self.attmix= torch.stack((w[f"blocks.{i}.att.time_mix_k"].squeeze(),
                  w[f"blocks.{i}.att.time_mix_v"].squeeze(),
                     w[f"blocks.{i}.att.time_mix_r"].squeeze())).unsqueeze(1).to(torch.float64).clone()
                
                self.time_first = w[f"blocks.{i}.att.time_first"].squeeze().to(torch.float64).clone()

                self.time_decay = w[f"blocks.{i}.att.time_decay"].squeeze().double().exp().neg().clone().to(torch.float64)

                # self.t = [powerTri(self.time_decay, i) for i in range(1, 21)]


                self.ffnmix= torch.stack((w[f"blocks.{i}.ffn.time_mix_k"].squeeze(),
                    w[f"blocks.{i}.ffn.time_mix_r"].squeeze())).unsqueeze(1).to(torch.float64).clone()
                

                torch.cuda.empty_cache()
            
            



            def forward(self, x, state):
                x = x.to(device=self.time_decay.device)
                xy = self.ln1(x)

                # tc = xy.roll(1, 0) not supported mps
                tc = xy[torch.arange(xy.shape[0])-1]
                rmc = tc[0].clone()
                tc[0] = state[0]
                state[0] = rmc

                # mix = torch.lerp(tc.unsqueeze(0), xy.unsqueeze(0), self.attmix) not supported mps
                mix = (tc.unsqueeze(0) * self.attmix + xy.unsqueeze(0) * (1-self.attmix))
              
                k,v,r = self.att(mix)


                r = r.sigmoid()

                # WKV kernel original

                wkv, state[2],state[3],state[4] = self.wkv(k.shape[0], k.shape[1], self.time_decay, self.time_first, k, v, state[2], state[3], state[4])
               
                wkv *= r

                rz = x + self.attout(wkv)

                ddd = self.ln2(rz)

                # rc = ddd.roll(1, 0)
                rc = ddd[torch.arange(ddd.shape[0])-1]
                dc = rc[0].clone()
                rc[0] = state[1]
                state[1] = dc

                # fmix = torch.lerp(rc, ddd, self.ffnmix)
                fmix = (rc * self.ffnmix + ddd * (1-self.ffnmix))

                

                rf = self.ffnreceptance(fmix[1]).sigmoid()
                
                kf = self.ffnkey(fmix[0]).relu()
                rvm = self.ffnvalue(torch.square(kf))

                out = rvm * rf + rz

                return out, state
            
            def config(self,i, **config):
                self.att.config(**config)
                self.ffnkey.config(**config)
                self.ffnvalue.config(**config)
                self.attout.config(**config)
                self.ffnreceptance.config(**config)
                self.wkv.config(**config)

                currentDevice = config["devices"][0]["device"]
                runtimedtype = torch.float32 if currentDevice == "mps" else torch.float64
                self.attmix = self.attmix.to(device=currentDevice, dtype=runtimedtype)
                self.time_first = self.time_first.to(device=currentDevice, dtype=runtimedtype)
                self.time_decay = self.time_decay.to(device=currentDevice, dtype=runtimedtype)
                self.ffnmix = self.ffnmix.to(currentDevice, dtype=runtimedtype)
                self.ln1.config(**config)
                self.ln2.config(**config)
                torch.cuda.empty_cache()

                
                
