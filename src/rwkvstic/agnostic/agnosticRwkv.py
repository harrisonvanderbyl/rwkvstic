from rwkvstic.agnostic.backends.base import module
from typing import Dict, List
import torch


def AgnosticRWKV(ops: module, path):

    device = "cuda"
    dtype = torch.float32

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

            self.atttime_mix_k = torch.nn.Parameter(
                torch.zeros(dims))
            self.atttime_mix_v = torch.nn.Parameter(
                torch.zeros(dims))
            self.atttime_mix_r = torch.nn.Parameter(
                torch.zeros(dims))

            self.time_first = torch.nn.Parameter(
                torch.zeros(dims))

            self.time_decay = torch.nn.Parameter(
                torch.zeros(dims))

            self.ffntime_mix_k = torch.nn.Parameter(
                torch.zeros(dims))
            self.ffntime_mix_r = torch.nn.Parameter(
                torch.zeros(dims))

            self.ffnkey = torch.nn.Linear(
                dims, dims*4, device=device)

            self.ffnvalue = torch.nn.Linear(
                dims, dims, device=device)

            self.ffnreceptance = torch.nn.Linear(
                dims, dims, device=device)

            self.attout = torch.nn.Linear(
                dims, dims, device=device)

        def loadFromBlinkDLCheckpoint(self, w, i):
            self.ln1.weight = torch.nn.Parameter(w[f"blocks.{i}.ln1.weight"])
            self.ln1.bias = torch.nn.Parameter(w[f"blocks.{i}.ln1.bias"])
            self.ln2.weight = torch.nn.Parameter(w[f"blocks.{i}.ln2.weight"])
            self.ln2.bias = torch.nn.Parameter(w[f"blocks.{i}.ln2.bias"])
            self.attkey.weight = torch.nn.Parameter(
                w[f"blocks.{i}.att.key.weight"])
            self.attvalue.weight = torch.nn.Parameter(
                w[f"blocks.{i}.att.value.weight"])
            self.ffnkey.weight = torch.nn.Parameter(
                w[f"blocks.{i}.ffn.key.weight"])
            self.ffnvalue.weight = torch.nn.Parameter(
                w[f"blocks.{i}.ffn.value.weight"])
            self.attout.weight = torch.nn.Parameter(
                w[f"blocks.{i}.att.output.weight"])
            self.ffnreceptance.weight = torch.nn.Parameter(
                w[f"blocks.{i}.ffn.receptance.weight"])
            self.attreceptance.weight = torch.nn.Parameter(
                w[f"blocks.{i}.att.receptance.weight"])

            self.atttime_mix_k = torch.nn.Parameter(
                w[f"blocks.{i}.att.time_mix_k"].squeeze())
            self.atttime_mix_v = torch.nn.Parameter(
                w[f"blocks.{i}.att.time_mix_v"].squeeze())
            self.atttime_mix_r = torch.nn.Parameter(
                w[f"blocks.{i}.att.time_mix_r"].squeeze())

            self.time_first = torch.nn.Parameter(
                w[f"blocks.{i}.att.time_first"].squeeze().exp())

            self.time_decay = torch.nn.Parameter(
                w[f"blocks.{i}.att.time_decay"].squeeze().double().exp().neg().exp().float())

            self.ffntime_mix_k = torch.nn.Parameter(
                w[f"blocks.{i}.ffn.time_mix_k"].squeeze())
            self.ffntime_mix_r = torch.nn.Parameter(
                w[f"blocks.{i}.ffn.time_mix_r"].squeeze())
            self.attreceptance.weight = torch.nn.Parameter(
                w[f"blocks.{i}.att.receptance.weight"])

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

        def processLayerx(self, k, v, rz: List[torch.Tensor], state, i: int):
            ki = k[i]

            state[2:4] = state[2:4] + torch.stack(
                (v[i], k[i]))

            # state[2:4] *= self.time_decay

            rz.append(state[2].clone())
            rz.append(state[3].clone())

            return rz, state

        def forward(self, x, state):

            xy = self.ln1(x)

            tc = xy.roll(1, 0)
            rmc = xy[-1]
            tc[0] = state[0]
            state[0] = rmc

            km = torch.lerp(tc, xy, self.atttime_mix_k)

            k = self.attkey(km).exp()

            vm = torch.lerp(tc, xy, self.atttime_mix_v)

            v = self.attvalue(vm)

            rm = torch.lerp(tc, xy, self.atttime_mix_r)

            r = self.attreceptance(rm).sigmoid()

            rz = []

            vx = k * v * \
                self.time_decay.pow(
                    1+len(k)-torch.arange(len(k)).cuda().unsqueeze(0).t())

            kx = k * \
                self.time_decay.pow(
                    1+len(k)-torch.arange(len(k)).cuda().unsqueeze(0).t())

            vx[0] += state[2]
            kx[0] += state[3]

            # for i in range(len(k)):
            #     rz, state = self.processLayerx(
            #         kx, v, rz, state, i)
            rza = vx.cumsum(0)
            rzb = kx.cumsum(0)
            state[2:4] = torch.stack((rza[-1], rzb[-1]))
            rz = (rza + k * self.time_first * v) / \
                (rzb + k * self.time_first)

            rz = self.attout(rz*r) + x

            ddd = self.ln2(rz)

            rc = ddd.roll(1, 0)
            dc = ddd[-1]
            rc[0] = state[1]
            state[1] = dc

            kmr = torch.lerp(rc, ddd, self.ffntime_mix_k)

            kf = self.ffnkey(kmr).relu()

            rmr = torch.lerp(rc, ddd, self.ffntime_mix_r)

            rf = self.ffnreceptance(rmr).sigmoid()

            rvm = self.ffnvalue(torch.square(kf))

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

                self.head = torch.nn.Linear(
                    dims, head, device=device)

                self.emb = torch.nn.Embedding(
                    head, dims)
                self.ln_out = torch.nn.LayerNorm(
                    dims, device=device, dtype=dtype)
                self.ln_in = torch.nn.LayerNorm(
                    dims, device=device, dtype=dtype)

                blocks = []
                for i in range(layers):
                    blocks.append(Block(dims))

                self.blocks = torch.nn.ModuleList(blocks)

        def loadFromBlinkDLCheckpoint(self, path):
            w = torch.load(path, map_location=device)
            with torch.no_grad():
                self.head.weight = torch.nn.Parameter(w["head.weight"])
                self.ln_in.bias = torch.nn.Parameter(w["blocks.0.ln0.bias"])
                self.ln_in.weight = torch.nn.Parameter(
                    w["blocks.0.ln0.weight"])

                self.ln_out.weight = torch.nn.Parameter(w["ln_out.weight"])
                self.ln_out.bias = torch.nn.Parameter(w["ln_out.bias"])
                self.emb.weight = torch.nn.Parameter(w["emb.weight"])

                for i in range(len(self.blocks)):
                    self.blocks[i].loadFromBlinkDLCheckpoint(w, i)

        def forward(self, x, state):
            x = self.emb(x.cuda())
            x = self.ln_in(x)

            for i, block in enumerate(self.blocks):

                x, rstate = block(x, state[i])
                state[i] = rstate

            x = self.ln_out(x)

            outx = self.head(x)[-1].detach()

            return outx, state
    myrwkv = myRWKV(768, 12, 50277)

    myrwkv.loadFromBlinkDLCheckpoint(path)

    myrwkv.eval()

    returnObject: myRWKV = myrwkv

    return returnObject
