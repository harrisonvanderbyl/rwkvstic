from rwkvstic.agnostic.backends.base import module
from typing import Dict, List
import torch


def AgnosticRWKV(ops: module, *args):

    class Block(torch.nn.Module):
        def __init__(self, w, i):
            super(Block, self).__init__()

            self.ln1 = torch.nn.LayerNorm(
                w[f"blocks.{i}.ln1.weight"].shape, device="cuda", dtype=ops.runtimedtype)
            self.ln2 = torch.nn.LayerNorm(
                w[f"blocks.{i}.ln2.weight"].shape, device="cuda", dtype=ops.runtimedtype)

            self.ln1.weight = torch.nn.Parameter(w[f"blocks.{i}.ln1.weight"])
            self.ln1.bias = torch.nn.Parameter(w[f"blocks.{i}.ln1.bias"])
            self.ln2.weight = torch.nn.Parameter(w[f"blocks.{i}.ln2.weight"])
            self.ln2.bias = torch.nn.Parameter(w[f"blocks.{i}.ln2.bias"])

            self.attkey = torch.nn.Linear(
                w[f"blocks.{i}.att.key.weight"].shape[0], w[f"blocks.{i}.att.key.weight"].shape[1], device="cuda")
            self.attkey.weight = torch.nn.Parameter(
                w[f"blocks.{i}.att.key.weight"].t())
            self.attvalue = torch.nn.Linear(
                w[f"blocks.{i}.att.value.weight"].shape[0], w[f"blocks.{i}.att.value.weight"].shape[1], device="cuda")
            self.attvalue.weight = torch.nn.Parameter(
                w[f"blocks.{i}.att.value.weight"].t())
            self.attreceptance = torch.nn.Linear(
                w[f"blocks.{i}.att.receptance.weight"].shape[0], w[f"blocks.{i}.att.receptance.weight"].shape[1], device="cuda")
            self.attreceptance.weight = torch.nn.Parameter(
                w[f"blocks.{i}.att.receptance.weight"].t())

            self.atttime_mix_k = torch.nn.Parameter(
                w[f"blocks.{i}.att.time_mix_k"])
            self.atttime_mix_v = torch.nn.Parameter(
                w[f"blocks.{i}.att.time_mix_v"])
            self.atttime_mix_r = torch.nn.Parameter(
                w[f"blocks.{i}.att.time_mix_r"])

            self.time_first = torch.nn.Parameter(
                w[f"blocks.{i}.att.time_first"])

            self.time_decay = torch.nn.Parameter(
                w[f"blocks.{i}.att.time_decay"])

            self.ffntime_mix_k = torch.nn.Parameter(
                w[f"blocks.{i}.ffn.time_mix_k"])
            self.ffntime_mix_r = torch.nn.Parameter(
                w[f"blocks.{i}.ffn.time_mix_r"])

            self.ffnkey = torch.nn.Linear(
                w[f"blocks.{i}.ffn.key.weight"].shape[0], w[f"blocks.{i}.ffn.key.weight"].shape[1], device="cuda")
            self.ffnkey.weight = torch.nn.Parameter(
                w[f"blocks.{i}.ffn.key.weight"].t())
            self.ffnvalue = torch.nn.Linear(
                w[f"blocks.{i}.ffn.value.weight"].shape[0], w[f"blocks.{i}.ffn.value.weight"].shape[1], device="cuda")
            self.ffnvalue.weight = torch.nn.Parameter(
                w[f"blocks.{i}.ffn.value.weight"].t())
            self.ffnreceptance = torch.nn.Linear(
                w[f"blocks.{i}.ffn.receptance.weight"].shape[0], w[f"blocks.{i}.ffn.receptance.weight"].shape[1], device="cuda")
            self.ffnreceptance.weight = torch.nn.Parameter(
                w[f"blocks.{i}.ffn.receptance.weight"].t())

            self.attout = torch.nn.Linear(
                w[f"blocks.{i}.att.output.weight"].shape[0], w[f"blocks.{i}.att.output.weight"].shape[1], device="cuda")
            self.attout.weight = torch.nn.Parameter(
                w[f"blocks.{i}.att.output.weight"].t())

        def processLayerx(self, k, v, rz: List[torch.Tensor], state, i):
            ww = self.time_first + k[i]
            p = torch.maximum(state[4], ww)

            e1 = (state[4] - p).exp()

            e2 = (ww - p).exp()

            a = e1 * (state[2]) + e2 * v[i]

            b = e1 * (state[3]) + e2

            wwn = state[4] + self.time_decay

            p1 = torch.maximum(wwn, k[i])

            e11 = (wwn - p1).exp()

            e21 = (k[i] - p1).exp()

            outb = e11 * state[2] + e21 * v[i]

            outc = e11 * state[3] + e21

            state[2:5] = torch.stack((outb, outc, p1))

            wkv = a / b

            rz.append(wkv)
            return rz, state

        def forward(self, x, state):

            xy = self.ln1(x)

            tc = xy.roll(1, 0)
            rmc = xy[-1]
            tc[0] = state[0]
            state[0] = rmc

            km = torch.lerp(tc, xy, self.atttime_mix_k)

            k = self.attkey(km)

            vm = torch.lerp(tc, xy, self.atttime_mix_v)

            v = self.attvalue(vm)

            rm = torch.lerp(tc, xy, self.atttime_mix_r)

            r = self.attreceptance(rm).sigmoid()

            rz = []

            for i in range(len(k)):
                rz, state = self.processLayerx(k, v, rz, state, i)

            rz = self.attout(torch.stack(rz)*r) + x

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

        def __init__(self, w):
            super(myRWKV, self).__init__()
            print("Legacy RWKV")

            for x in ops.__dict__.keys():
                self.__dict__[x] = ops.__dict__[x]

            self.head = torch.nn.Linear(
                w["head.weight"].shape[0], w["head.weight"].shape[1], device="cuda")
            self.head.weight = torch.nn.Parameter(w["head.weight"].t())

            self.emb = torch.nn.Embedding(
                w["emb.weight"].shape[0], w["emb.weight"].shape[1])
            self.ln_out = torch.nn.LayerNorm(
                w["ln_out.weight"].shape, device="cuda", dtype=ops.runtimedtype)
            self.ln_in = torch.nn.LayerNorm(
                w["blocks.0.ln0.weight"].shape, device="cuda", dtype=ops.runtimedtype)

            self.ln_in.weight = torch.nn.Parameter(w["blocks.0.ln0.weight"])
            self.ln_in.bias = torch.nn.Parameter(w["blocks.0.ln0.bias"])

            self.ln_out.weight = torch.nn.Parameter(w["ln_out.weight"])
            self.ln_out.bias = torch.nn.Parameter(w["ln_out.bias"])

            blocks = []
            for i in range(ops.n_layers):
                blocks.append(Block(w, i))

            self.blocks = torch.nn.ModuleList(blocks)

            self.emb.weight = torch.nn.Parameter(w["emb.weight"])

        # def processLayer(self, k, v, rz: List[torch.Tensor], state, xx: int, i: int):
        #     ki = self.exp(k[i])
        #     wrd = self.divide(
        #         self.add(state[2], self.multiply(self.multiply(ki, v[i]), self.exp(self.time_first[xx]))), self.add(state[3], self.multiply(ki, self.exp(self.time_first[xx]))))

        #     state = self.scatter(state, self.scatterindices[1], self.multiply(self.exp(self.time_decay[xx]), self.add(
        #         state[2:4], self.stack((self.multiply(
        #             v[i], ki), ki)))))

        #     rz = self.arrayPush(rz, wrd, i)
        #     return rz, state

        def forward(self, x, state):
            x = self.emb(x).cuda()
            x = self.ln_in(x)

            for i in range(len(self.blocks)):

                x, rstate = self.blocks[i](x, state[i])
                state[i] = rstate

            x = self.ln_out(x)

            outx = self.head(x)

            return outx[-1].detach(), state

        # for keras stuff, ignore this if you are not using keras
        def call(self, *args, **kwds):
            del kwds["training"]
            return self.forward(*args, **kwds)
    returnObject: myRWKV = ops.postProcessModule(myRWKV(*args))
    return returnObject
