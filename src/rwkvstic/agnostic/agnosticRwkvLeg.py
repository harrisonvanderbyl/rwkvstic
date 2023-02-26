from rwkvstic.agnostic.backends.base import module
from typing import Dict


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

            tc = self.roll(xy)
            tc[0] = statea

            k = self.matvec(
                self.key[xx], self.lerp(tc, xy, self.kktk[xx]))

            v = self.matvec(self.value[xx], self.lerp(
                tc, xy, self.vvtv[xx]))
            rr = self.matvec(
                self.receptance[xx], self.lerp(tc, xy, self.rrtr[xx]))
            r = self.logistical(rr)

            ww = self.add(k, self.time_first[xx])
            rz = []
            for i in range(self.len(x)):
                p = self.maximum(statee, ww[i])

                e1 = self.exp(self.subtract(statee, p))
                e2 = self.exp(self.subtract(ww[i], p))
                a = self.add(self.multiply(e1, stateb),
                             self.multiply(e2, v[i]))
                b = self.add(self.multiply(e1, statec), e2)
                wwn = self.add(statee, self.time_decay[xx])

                p = self.maximum(wwn, k[i])

                e1 = self.exp(self.subtract(wwn, p))
                e2 = self.exp(self.subtract(k[i], p))
                outb = self.add(self.multiply(e1, stateb),
                                self.multiply(e2, v[i]))
                outc = self.add(self.multiply(e1, statec), e2)
                eee = p
                wkv = self.divide(a, b)
                rz.append(wkv)
                statee = eee
                stateb = outb
                statec = outc

            mvv = self.add(x, self.matvec(
                self.outputvv[xx], self.multiply(r, self.stack(rz))))

            ddd = self.layernorm(mvv, self.ln2w[xx], self.ln2b[xx])

            rc = self.roll(ddd)
            rc[0] = stated

            km = self.relu(self.matvec(self.key_ffn[xx], self.lerp(
                rc, ddd, self.time_mix_k_ffn[xx])))

            rt = self.logistical((self.matvec(self.receptance_ffn[xx], self.lerp(
                rc, ddd, self.time_mix_r_ffn[xx]))))

            x = self.add(mvv, self.multiply(
                self.matvec(self.value_ffn[xx], self.multiply(km, km)), rt))

            return x, xy[-1], stateb, statec, ddd[-1], statee

        @ ops.mainfunc
        def forward(self, x, state):

            x = self.layernorm(
                self.processEmbed(self.getIndex(self.emb, x)),
                self.emb1, self.emb2)

            statea = state[0::5]
            stateb = state[1::5]
            statec = state[2::5]
            stated = state[3::5]
            statee = state[4::5]

            ot = []

            for i in range(self.n_layers):
                x, aaa, bbb, ccc, ddd, eee = self.doLayer(
                    x, statea[i], stateb[i], statec[i], stated[i], statee[i], i)

                ot = ot + [aaa, bbb, ccc, ddd, eee]

            x = self.matvec(self.postprocess2, self.layernorm(x, self.postprocess0,
                                                              self.postprocess1))

            return self.postProcessTensor(x[-1]), self.stack(ot)

        # for keras stuff, ignore this if you are not using keras
        def call(self, *args, **kwds):
            del kwds["training"]
            return self.forward(*args, **kwds)
    returnObject: myRWKV = ops.postProcessModule(myRWKV(*args))
    return returnObject
