from rwkvstic.agnostic.backends.base import module
from typing import Dict


def LegacyRWKV(ops: module, *args):
    class myRWKV(ops.module):

        @ ops.initfunc
        def __init__(self, w: Dict[str, ops.TensorType]):
            super(myRWKV, self).__init__()
            print("Legacy RWKV")

            self.ops = ops
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
        def doLayer(self, x, statea, stateb, statec, stated, statee, xx):

            xy = ops.stack(
                [ops.layernorm(y, self.ln1w[xx], self.ln1b[xx]) for y in x])
            ct = ops.cat([statea.unsqueeze(0), xy[:-1]])

            k = ops.matvec(
                self.key[xx], ops.lerp(ct, xy, self.kktk[xx]))

            v = ops.matvec(self.value[xx], ops.lerp(
                ct, xy, self.vvtv[xx]))
            rr = ops.matvec(
                self.receptance[xx], ops.lerp(ct, xy, self.rrtr[xx]))
            r = ops.logistical(rr)

            ww = ops.add(k, self.time_first[xx])
            rz = []
            for i in range(len(x)):
                p = ops.maximum(statee, ww[i])

                e1 = ops.exp(ops.subtract(statee, p))
                e2 = ops.exp(ops.subtract(ww[i], p))
                a = ops.add(ops.multiply(e1, stateb), ops.multiply(e2, v[i]))
                b = ops.add(ops.multiply(e1, statec), e2)
                wwn = ops.add(statee, self.time_decay[xx])

                p = ops.maximum(wwn, k[i])

                e1 = ops.exp(ops.subtract(wwn, p))
                e2 = ops.exp(ops.subtract(k[i], p))
                outb = ops.add(ops.multiply(e1, stateb),
                               ops.multiply(e2, v[i]))
                outc = ops.add(ops.multiply(e1, statec), e2)
                eee = p
                wkv = ops.divide(a, b)
                rz.append(wkv)
                statee = eee
                stateb = outb
                statec = outc

            mvv = ops.add(x, ops.matvec(
                self.outputvv[xx], ops.multiply(r, ops.stack(rz))))

            ddd = ops.stack(
                [ops.layernorm(y, self.ln2w[xx], self.ln2b[xx]) for y in mvv]
            )

            ctt = ops.cat([stated.unsqueeze(0), ddd[:-1]])

            km = ops.relu(ops.matvec(self.key_ffn[xx], ops.lerp(
                ctt, ddd, self.time_mix_k_ffn[xx])))

            rt = ops.logistical((ops.matvec(self.receptance_ffn[xx], ops.lerp(
                ctt, ddd, self.time_mix_r_ffn[xx]))))

            x = ops.add(mvv, ops.multiply(
                ops.matvec(self.value_ffn[xx], ops.multiply(km, km)), rt))

            return x, xy[-1], outb, outc, ddd[-1], eee

        @ ops.mainfunc
        def forward(self, x: ops.VectorType, state: ops.MatrixType = None):

            if (state is None):
                state = ops.emptyState

            x = ops.stack([ops.layernorm(
                ops.processEmbed(z),
                self.emb1, self.emb2) for z in ops.getIndex(self.emb, x)])

            statea = state[0::5]
            stateb = state[1::5]
            statec = state[2::5]
            stated = state[3::5]
            statee = state[4::5]

            ot = []

            for i in range(ops.n_layers):
                x, aaa, bbb, ccc, ddd, eee = self.doLayer(
                    x, statea[i], stateb[i], statec[i], stated[i], statee[i], i)

                ot = ot + [aaa, bbb, ccc, ddd, eee]

            x = ops.matvec(self.postprocess2, ops.layernorm(x, self.postprocess0,
                                                            self.postprocess1))

            return ops.postProcessTensor(x[-1]), ops.stack(ot)

        # for keras stuff, ignore this if you are not using keras
        def call(self, *args, **kwds):
            del kwds["training"]
            return self.forward(*args, **kwds)
    returnObject: myRWKV = ops.postProcessModule(myRWKV(*args))
    return returnObject
