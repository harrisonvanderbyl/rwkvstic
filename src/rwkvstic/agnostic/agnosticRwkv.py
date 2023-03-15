from rwkvstic.agnostic.backends.base import module
from typing import Dict, List
import torch
# set torch to use all 12 cores on amd
torch.set_num_threads(24)
# set torch to use hardware specific optimizations
torch.backends.cudnn.benchmark = True
# set torch to use deterministic algorithms
torch.backends.cudnn.deterministic = True
# set torch to use deterministic algorithms
torch.backends.cudnn.enabled = True


def AgnosticRWKV(ops: module, path):

    device = "cuda" if ops.useGPU else "cpu"

    dtype = ops.dtype
    runtimedtype = ops.runtimedtype

    def powerTri(t, p):
        t = t.cpu().expand(p, p, -1)

        tri = ((torch.arange(p).expand(p, p)+1).t() -
               torch.arange(p)).tril().unsqueeze(-1)

        mask = torch.ones(p, p).tril().unsqueeze(-1).to(torch.bool)

        return ((t*tri).exp()*mask).to(device).to(runtimedtype)

    class Block(torch.nn.Module):
        def __init__(self, dims):
            super(Block, self).__init__()

            self.dtype = dtype
            self.runtimedtype = runtimedtype
            with torch.no_grad():
                self.ln1 = torch.nn.LayerNorm(
                    (dims,), device=device, dtype=runtimedtype)
                self.ln2 = torch.nn.LayerNorm(
                    dims, device=device, dtype=runtimedtype)

                self.attkey = torch.nn.Linear(
                    dims, dims, device=device, dtype=dtype)

                self.attvalue = torch.nn.Linear(
                    dims, dims, device=device, dtype=dtype)

                self.attreceptance = torch.nn.Linear(
                    dims, dims, device=device, dtype=dtype)

                self.ffnkey = torch.nn.Linear(
                    dims, dims*4, device=device, dtype=dtype)

                self.ffnvalue = torch.nn.Linear(
                    dims, dims, device=device, dtype=dtype)

                self.ffnreceptance = torch.nn.Linear(
                    dims, dims, device=device, dtype=dtype)

                self.attout = torch.nn.Linear(
                    dims, dims, device=device, dtype=dtype)

                self.t = []

        def loadFromBlinkDLCheckpoint(self, w, i):
            with torch.no_grad():
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
                    w[f"blocks.{i}.att.time_first"].squeeze().to(runtimedtype))

                self.time_decay = torch.nn.Parameter(
                    w[f"blocks.{i}.att.time_decay"].squeeze().to(runtimedtype).double().exp().neg())

                self.t = [powerTri(self.time_decay, i) for i in range(1, 21)]

                self.ffntime_mix_k = torch.nn.Parameter(
                    w[f"blocks.{i}.ffn.time_mix_k"].squeeze().to(runtimedtype))
                self.ffntime_mix_r = torch.nn.Parameter(
                    w[f"blocks.{i}.ffn.time_mix_r"].squeeze().to(runtimedtype))
                self.attreceptance.weight = torch.nn.Parameter(
                    w[f"blocks.{i}.att.receptance.weight"].to(dtype))

        def forward(self, x, state):

            xy = self.ln1(x)

            tc = xy.roll(1, 0)
            rmc = xy[-1]
            tc[0] = state[0]
            state[0] = rmc

            km = torch.lerp(tc, xy, self.atttime_mix_k)

            k = self.attkey(km.to(self.dtype)).to(self.runtimedtype)

            vm = torch.lerp(tc, xy, self.atttime_mix_v)

            v = self.attvalue(vm.to(self.dtype)).to(self.runtimedtype)

            rm = torch.lerp(tc, xy, self.atttime_mix_r)

            r = self.attreceptance(rm.to(self.dtype)).to(
                self.runtimedtype).sigmoid()

            vx_kx = (k).exp().unsqueeze(0) .expand(
                2, k.shape[0], k.shape[1]).clone()
            vx_kx[0] *= v

            t = self.t[k.shape[0]-1]
            vx_kx[0][0] += state[2]
            vx_kx[1][0] += state[3]

            rza = torch.einsum("rki,jki->rji", vx_kx, t)
            # vx_kx[0][0] += state[2]
            # vx_kx[1][0] += state[3]
            # rza = (vx_kx*t.unsqueeze(0)).sum(2)
            vx_kx *= self.time_first.exp()
            vx_kx += rza
            vx_kx[0] = r*vx_kx[0]
            vx_kx[1] = 1/vx_kx[1]
            wkv = vx_kx.prod(0)

            state[2] = rza[0][-1]
            state[3] = rza[1][-1]

            rz = x + self.attout((wkv).to(self.dtype)).to(self.runtimedtype)

            ddd = self.ln2(rz)

            rc = ddd.roll(1, 0)
            dc = ddd[-1]
            rc[0] = state[1]
            state[1] = dc

            kmr = torch.lerp(rc, ddd, self.ffntime_mix_k)

            kf = self.ffnkey(kmr.to(self.dtype)).to(self.runtimedtype).relu()

            rmr = torch.lerp(rc, ddd, self.ffntime_mix_r)

            rf = self.ffnreceptance(rmr.to(self.dtype)).to(
                self.runtimedtype).sigmoid()

            rvm = self.ffnvalue(torch.square(kf).to(
                self.dtype)).to(self.runtimedtype)

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

                self.dtype = dtype
                self.runtimedtype = runtimedtype

        def loadFromBlinkDLCheckpoint(self, path):

            w = torch.load(path, map_location=device)

            dims = w["blocks.0.att.key.weight"].shape[0]
            head = 50277
            layers = len(
                list(filter(lambda x: "blocks" in x and "ln1.bias" in x, w.keys())))

            ops.emptyState = torch.zeros(
                (layers, 4, dims), device=device, dtype=runtimedtype)

            with torch.no_grad():

                self.head = torch.nn.Linear(
                    dims, head, device=device, dtype=dtype,)

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

                self.device = device

        def forward(self, x, state):
            x = self.emb(x.to(self.device))
            x = self.ln_in(x)

            for i, block in enumerate(self.blocks):

                x, rstate = block(x, state[i])
                state[i] = rstate

            x = self.ln_out(x).to(self.dtype)

            outx = self.head(x)[-1].detach()

            return outx, state
    with torch.no_grad():
        myrwkv = myRWKV(768, 12, 50277)

        myrwkv.loadFromBlinkDLCheckpoint(path)

        myrwkv.eval()

        # myrwkv = torch.jit.script(myrwkv)

        returnObject: myRWKV = myrwkv

        return returnObject
