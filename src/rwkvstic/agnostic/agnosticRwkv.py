from rwkvstic.agnostic.backends.base import module
from typing import Dict, List
import torch





def AgnosticRWKV(ops: module, path):

    device = "cuda" if ops.useGPU else "cpu"
    
    dtype = torch.float32
    runtimedtype = torch.float64

    def powerTri(t, p):
        t = t.expand(p, p, -1)

        tri = ((torch.arange(p).expand(p, p)+1).t() -
            torch.arange(p)).tril().unsqueeze(-1).to(device)

        mask = torch.ones(p, p).tril().unsqueeze(-1).to(device)

        return t.pow(tri)*mask

    class Block(torch.nn.Module):
        def __init__(self, dims):
            super(Block, self).__init__()

            self.ln1 = torch.nn.LayerNorm(
                (dims,), device=device, dtype=dtype)
            self.ln2 = torch.nn.LayerNorm(
                dims, device=device, dtype=dtype)

            self.attkey = torch.nn.Linear(
                dims, dims, device=device)

            self.attvalue = torch.nn.Linear(
                dims, dims, device=device)

            self.attreceptance = torch.nn.Linear(
                dims, dims, device=device)

            self.ffnkey = torch.nn.Linear(
                dims, dims*4, device=device)

            self.ffnvalue = torch.nn.Linear(
                dims, dims, device=device)

            self.ffnreceptance = torch.nn.Linear(
                dims, dims, device=device)

            self.attout = torch.nn.Linear(
                dims, dims, device=device)

            self.t = []

        def loadFromBlinkDLCheckpoint(self, w, i):
            self.ln1.weight = torch.nn.Parameter(
                w[f"blocks.{i}.ln1.weight"].to(runtimedtype))
            self.ln1.bias = torch.nn.Parameter(
                w[f"blocks.{i}.ln1.bias"].to(runtimedtype))
            self.ln2.weight = torch.nn.Parameter(
                w[f"blocks.{i}.ln2.weight"].to(runtimedtype))
            self.ln2.bias = torch.nn.Parameter(
                w[f"blocks.{i}.ln2.bias"].to(runtimedtype))
            self.attkey.weight = torch.nn.Parameter(
                w[f"blocks.{i}.att.key.weight"].to(dtype))
            self.attvalue.weight = torch.nn.Parameter(
                w[f"blocks.{i}.att.value.weight"].to(dtype))
            self.ffnkey.weight = torch.nn.Parameter(
                w[f"blocks.{i}.ffn.key.weight"].to(dtype))
            self.ffnvalue.weight = torch.nn.Parameter(
                w[f"blocks.{i}.ffn.value.weight"].to(dtype))
            self.attout.weight = torch.nn.Parameter(
                w[f"blocks.{i}.att.output.weight"].to(dtype))
            self.ffnreceptance.weight = torch.nn.Parameter(
                w[f"blocks.{i}.ffn.receptance.weight"].to(dtype))
            self.attreceptance.weight = torch.nn.Parameter(
                w[f"blocks.{i}.att.receptance.weight"].to(dtype))

            self.atttime_mix_k = torch.nn.Parameter(
                w[f"blocks.{i}.att.time_mix_k"].squeeze().to(runtimedtype))
            self.atttime_mix_v = torch.nn.Parameter(
                w[f"blocks.{i}.att.time_mix_v"].squeeze().to(runtimedtype))
            self.atttime_mix_r = torch.nn.Parameter(
                w[f"blocks.{i}.att.time_mix_r"].squeeze().to(runtimedtype))

            self.time_first = torch.nn.Parameter(
                w[f"blocks.{i}.att.time_first"].squeeze().to(runtimedtype).exp())

            self.time_decay = torch.nn.Parameter(
                w[f"blocks.{i}.att.time_decay"].squeeze().to(runtimedtype).double().exp().neg().exp())

            self.t = [powerTri(self.time_decay, i) for i in range(1, 21)]

            self.ffntime_mix_k = torch.nn.Parameter(
                w[f"blocks.{i}.ffn.time_mix_k"].squeeze().to(runtimedtype))
            self.ffntime_mix_r = torch.nn.Parameter(
                w[f"blocks.{i}.ffn.time_mix_r"].squeeze().to(runtimedtype))
            self.attreceptance.weight = torch.nn.Parameter(
                w[f"blocks.{i}.att.receptance.weight"].to(dtype))

        # def processLayer(self, k, v, rz: List[torch.Tensor], state, i: int):
        #     ww = self.time_first + k[i]
        #     p = torch.maximum(state[4], ww)

        #     e1 = (state[4] - p).exp()

        #     e2 = (ww - p).exp()

        #     a = e1 * (state[2]) + e2 * v[i]

        #     b = e1 * (state[3]) + e2

        #     wwn = state[4] + self.time_decay

        #     p1 = torch.maximum(wwn, k[i])

        #     e11 = (wwn - p1).exp()

        #     e21 = (k[i] - p1).exp()

        #     outb = e11 * state[2] + e21 * v[i]

        #     outc = e11 * state[3] + e21

        #     state[2:5] = torch.stack((outb, outc, p1))

        #     wkv = a / b

        #     rz.append(wkv)
        #     return rz, state

        # def processLayerx(self, k, v, rz: List[torch.Tensor], state, i: int):
        #     ki = k[i]

        #     state[2:4] = (state[2:4] + torch.stack(
        #         (v[i], k[i])))*self.time_decay

        #     # state[2:4] *= self.time_decay

        #     rz.append(state[2].clone())
        #     rz.append(state[3].clone())

        #     return rz, state

        def forward(self, x, state):

            xy = self.ln1(x)

            tc = xy.roll(1, 0)
            rmc = xy[-1]
            tc[0] = state[0]
            state[0] = rmc

            km = torch.lerp(tc, xy, self.atttime_mix_k)

            k = self.attkey(km.to(dtype)).to(runtimedtype)

            vm = torch.lerp(tc, xy, self.atttime_mix_v)

            v = self.attvalue(vm.to(dtype)).to(runtimedtype)

            rm = torch.lerp(tc, xy, self.atttime_mix_r)

            r = self.attreceptance(rm.to(dtype)).to(runtimedtype).sigmoid()

            rz = []

            k = k.exp()

            vx = k * v

            kx = k

            t = self.t[k.shape[0]-1]

            vx[0] += state[2]
            kx[0] += state[3]

            rza = (vx*t).sum(1)
            rzb = (kx*t).sum(1)

            state[2] = rza[-1]
            state[3] = rzb[-1]

            rz = (rza + vx * self.time_first) / \
                (rzb + kx * self.time_first)

            rz = x + self.attout((rz*r).to(dtype)).to(runtimedtype)

            ddd = self.ln2(rz)

            rc = ddd.roll(1, 0)
            dc = ddd[-1]
            rc[0] = state[1]
            state[1] = dc

            kmr = torch.lerp(rc, ddd, self.ffntime_mix_k)

            kf = self.ffnkey(kmr.to(dtype)).to(runtimedtype).relu()

            rmr = torch.lerp(rc, ddd, self.ffntime_mix_r)

            rf = self.ffnreceptance(rmr.to(dtype)).to(runtimedtype).sigmoid()

            rvm = self.ffnvalue(torch.square(kf).to(dtype)).to(runtimedtype)

            out = rvm * rf + rz

            return out, state

            # stuff

    class myRWKV(torch.nn.Module):

        def __init__(self, dims, layers, head):
            super(myRWKV, self).__init__()
            print("Legacy RWKV")
            with torch.no_grad():
                for x in ops.__dict__.keys():
                    self.__dict__[x] = ops.__dict__[x]

                

                

        def loadFromBlinkDLCheckpoint(self, path):


            w = torch.load(path, map_location=device)

            dims = w["blocks.0.att.key.weight"].shape[0]
            head = 50277
            layers = len(list(filter(lambda x: "blocks" in x and "ln1.bias" in x, w.keys())))

            ops.emptyState = torch.zeros((layers,5, dims), device=device, dtype=dtype)


            with torch.no_grad():

                self.head = torch.nn.Linear(
                    dims, head, device=device)

                self.emb = torch.nn.Embedding(
                    head, dims)
                self.ln_out = torch.nn.LayerNorm(
                    dims, device=device, dtype=dtype)
                self.ln_in = torch.nn.LayerNorm(
                    dims, device=device, dtype=dtype)

                self.head.weight = torch.nn.Parameter(
                    w["head.weight"].to(dtype))
                self.ln_in.bias = torch.nn.Parameter(
                    w["blocks.0.ln0.bias"].to(runtimedtype))
                self.ln_in.weight = torch.nn.Parameter(
                    w["blocks.0.ln0.weight"].to(runtimedtype))

                self.ln_out.weight = torch.nn.Parameter(
                    w["ln_out.weight"].to(runtimedtype))
                self.ln_out.bias = torch.nn.Parameter(
                    w["ln_out.bias"].to(runtimedtype))
                self.emb.weight = torch.nn.Parameter(
                    w["emb.weight"].to(runtimedtype))
                
                blocks = []
                for i in range(layers):
                    blocks.append(Block(dims))

                self.blocks = torch.nn.ModuleList(blocks)

                for i in range(len(self.blocks)):

                    self.blocks[i].loadFromBlinkDLCheckpoint(w, i)

        def forward(self, x, state):
            x = self.emb(x.to(device))
            x = self.ln_in(x)

            for i, block in enumerate(self.blocks):

                x, rstate = block(x, state[i])
                state[i] = rstate

            x = self.ln_out(x).to(dtype)

            outx = self.head(x)[-1].detach()

            return outx, state
    with torch.no_grad():
        myrwkv = myRWKV(768, 12, 50277)

        myrwkv.loadFromBlinkDLCheckpoint(path)

        myrwkv.eval()
        

        myrwkv = myrwkv

        returnObject: myRWKV = myrwkv

        return returnObject
