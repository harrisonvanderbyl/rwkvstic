from rwkvstic.agnostic.backends.base import module
from typing import Dict
import tensorflow as tf


def LegacyRWKV(ops: module, *args):
    class myRWKV(ops.module):

        @ ops.initfunc
        def __init__(self, w: Dict[str, ops.TensorType]):
            super(myRWKV, self).__init__()
            print("Legacy RWKV")

            for x in ops.__dict__.keys():
                self.__dict__[x] = ops.__dict__[x]
            self.postprocess0: ops.VectorType = (w["ln_out.weight"])
            self.postprocess1: ops.VectorType = (w["ln_out.bias"])
            self.postprocess2: ops.VectorType = (w["head.weight"])
            self.emb: ops.MatrixType = w["emb.weight"]
            self.emb1: ops.VectorType = w["blocks.0.ln0.weight"]
            self.emb2: ops.VectorType = w["blocks.0.ln0.bias"]
            self.ln1w: ops.VectorType = ops.stack(
                [w[f"blocks.{x}.ln1.weight"] for x in range(ops.n_layers)])
            self.ln1b: ops.VectorType = ops.stack(
                [w[f"blocks.{x}.ln1.bias"] for x in range(ops.n_layers)])
            self.ln2w: ops.VectorType = ops.stack(
                [w[f"blocks.{x}.ln2.weight"] for x in range(ops.n_layers)])
            self.ln2b: ops.VectorType = ops.stack(
                [w[f"blocks.{x}.ln2.bias"] for x in range(ops.n_layers)])
            self.time_decay: ops.VectorType = ops.stack([
                w[f"blocks.{x}.att.time_decay"] for x in range(ops.n_layers)])
            self.time_first: ops.VectorType = ops.stack([
                w[f"blocks.{x}.att.time_first"] for x in range(ops.n_layers)])
            self.kktk: ops.VectorType = ops.stack(
                [w[f"blocks.{x}.att.time_mix_k"] for x in range(ops.n_layers)])
            self.vvtv: ops.VectorType = ops.stack(
                [w[f"blocks.{x}.att.time_mix_v"] for x in range(ops.n_layers)])
            self.rrtr: ops.VectorType = ops.stack(
                [w[f"blocks.{x}.att.time_mix_r"] for x in range(ops.n_layers)])
            self.key: ops.MatrixType = ops.stack(
                [w[f"blocks.{x}.att.key.weight"] for x in range(ops.n_layers)])
            self.value: ops.MatrixType = ops.stack(
                [w[f"blocks.{x}.att.value.weight"] for x in range(ops.n_layers)])
            self.receptance: ops.MatrixType = ops.stack([
                w[f"blocks.{x}.att.receptance.weight"] for x in range(ops.n_layers)])
            self.outputvv: ops.MatrixType = ops.stack([
                w[f"blocks.{x}.att.output.weight"] for x in range(ops.n_layers)])
            self.time_mix_k_ffn: ops.VectorType = ops.stack([
                w[f"blocks.{x}.ffn.time_mix_k"] for x in range(ops.n_layers)])
            self.time_mix_r_ffn: ops.VectorType = ops.stack([
                w[f"blocks.{x}.ffn.time_mix_r"] for x in range(ops.n_layers)])
            self.key_ffn: ops.MatrixType = ops.stack(
                [w[f"blocks.{x}.ffn.key.weight"] for x in range(ops.n_layers)])
            self.receptance_ffn: ops.MatrixType = ops.stack([
                w[f"blocks.{x}.ffn.receptance.weight"] for x in range(ops.n_layers)])
            self.value_ffn: ops.MatrixType = ops.stack([
                w[f"blocks.{x}.ffn.value.weight"] for x in range(ops.n_layers)])

        @ops.layerdef
        def doLayer(self, x, statea, stateb, statec, stated, statee, xx: int):

            xy = self.layernorm(x, self.ln1w[xx], self.ln1b[xx])

            tc = self.push(self.roll(xy), statea)

            k = self.matvec(
                self.key[xx], self.lerp(tc, xy, self.kktk[xx]))

            v = self.matvec(self.value[xx], self.lerp(
                tc, xy, self.vvtv[xx]))

            rr = self.matvec(
                self.receptance[xx], self.lerp(tc, xy, self.rrtr[xx]))

            r = self.logistical(rr)

            ww = self.add(k, self.time_first[xx])
            rz = self.emptyarray(self.len(x))
            bz = self.emptyarray(self.len(x)+1)
            cz = self.emptyarray(self.len(x)+1)
            gz = self.emptyarray(self.len(x)+1)
            bz = self.arrayPush(bz, statee, 0)
            cz = self.arrayPush(cz, stateb, 0)
            gz = self.arrayPush(gz, statec, 0)

            for i in self.rng(self.len(x)):

                p = self.maximum(self.arrayGet(bz, i), ww[i])

                e1 = self.exp(self.subtract(self.arrayGet(bz, i), p))

                e2 = self.exp(self.subtract(ww[i], p))

                a = self.add(self.multiply(e1, self.arrayGet(cz, i)),
                             self.multiply(e2, v[i]))

                b = self.add(self.multiply(e1, self.arrayGet(gz, i)), e2)

                wwn = self.add(self.arrayGet(bz, i), self.time_decay[xx])

                p1 = self.maximum(wwn, k[i])

                e11 = self.exp(self.subtract(wwn, p1))

                e21 = self.exp(self.subtract(k[i], p1))

                outb = self.add(self.multiply(e11, self.arrayGet(cz, i)),
                                self.multiply(e21, v[i]))

                outc = self.add(self.multiply(e11, self.arrayGet(gz, i)), e21)

                wkv = self.divide(a, b)
                rz = self.arrayPush(rz, wkv, i)
                bz = self.arrayPush(bz, p1, i+1)
                cz = self.arrayPush(cz, outb, i+1)

                gz = self.arrayPush(gz, outc, i+1)

            mvv = self.add(x, self.matvec(
                self.outputvv[xx], self.multiply(r, self.stack(rz))))

            ddd = self.layernorm(mvv, self.ln2w[xx], self.ln2b[xx])

            rc = self.push(self.roll(ddd), stated)

            km = self.relu(self.matvec(self.key_ffn[xx], self.lerp(
                rc, ddd, self.time_mix_k_ffn[xx])))

            rt = self.logistical((self.matvec(self.receptance_ffn[xx], self.lerp(
                rc, ddd, self.time_mix_r_ffn[xx]))))

            rvm = self.matvec(self.value_ffn[xx], self.multiply(km, km))

            x = self.add(mvv, self.multiply(
                rvm, rt))

            return x,  self.pop(xy), self.arrayGet(cz, self.len(x)), self.arrayGet(gz, self.len(x)),  self.pop(ddd), self.arrayGet(bz, self.len(x))

        @ ops.mainfunc
        def forward(self, x, state):
            g = self.getIndex(self.emb, x)
            x = self.layernorm(
                self.processEmbed(g),
                self.emb1, self.emb2)

            statea = state[0::5]
            stateb = state[1::5]
            statec = state[2::5]
            stated = state[3::5]
            statee = state[4::5]

            ot = self.emptyarray(self.n_layers*5)

            for i in self.rng(self.n_layers):

                x, aaa, bbb, ccc, ddd, eee = self.doLayer(
                    x, statea[i], stateb[i], statec[i], stated[i], statee[i], i)

                ot = self.arrayPush(ot, aaa, i*5)
                ot = self.arrayPush(ot, bbb, i*5+1)
                ot = self.arrayPush(ot, ccc, i*5+2)
                ot = self.arrayPush(ot, ddd, i*5+3)
                ot = self.arrayPush(ot, eee, i*5+4)

            x = self.matvec(self.postprocess2, self.layernorm(x, self.postprocess0,
                                                              self.postprocess1))

            return self.postProcessTensor(self.pop(x)), self.stack(ot)

        # for keras stuff, ignore this if you are not using keras
        def call(self, *args, **kwds):
            del kwds["training"]
            return self.forward(*args, **kwds)
    returnObject: myRWKV = ops.postProcessModule(myRWKV(*args))
    return returnObject
