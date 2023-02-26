import torch
from rwkvstic.agnostic.backends.base import module
from typing import Dict


def AgnostigRWKV(ops: module, *args):
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
        def doLayer(self, x, statea, stateb, statec, stated, xx: int):

            xy = self.layernorm(x, self.ln1w[xx], self.ln1b[xx])
            ct = self.cat([self.unsqueeze(statea, 0), xy[:-1]])

            kk = self.matvec(
                self.key[xx], self.lerp(ct, xy, self.kktk[xx]))

            v = self.matvec(self.value[xx], self.lerp(
                ct, xy, self.vvtv[xx]))
            rr = self.matvec(
                self.receptance[xx], self.lerp(ct, xy, self.rrtr[xx]))
            r = self.logistical(rr)
            k = ops.exp(kk)
            rz = []
            for i in range(self.len(x)):

                wrd = ops.divide(
                    ops.add(stateb, ops.multiply(ops.multiply(k[i], v[i]), self.time_first[xx])), ops.add(statec, ops.multiply(k[i], self.time_first[xx])))

                stateb = ops.multiply(self.time_decay[xx], ops.add(
                    stateb, ops.multiply(k[i], v[i])))
                statec = ops.multiply(
                    ops.add(statec, k[i]), self.time_decay[xx])

                rz += [wrd]
            mvv = self.add(x, self.matvec(
                self.outputvv[xx], self.multiply(r, self.stack(rz))))

            ddd = self.layernorm(mvv, self.ln2w[xx], self.ln2b[xx])

            ctt = self.cat([self.unsqueeze(stated, 0), ddd[:-1]])

            km = self.relu(self.matvec(self.key_ffn[xx], self.lerp(
                ctt, ddd, self.time_mix_k_ffn[xx])))

            rt = self.logistical((self.matvec(self.receptance_ffn[xx], self.lerp(
                ctt, ddd, self.time_mix_r_ffn[xx]))))

            x = self.add(mvv, self.multiply(
                self.matvec(self.value_ffn[xx], self.multiply(km, km)), rt))

            return x, xy[-1], stateb, statec, ddd[-1]

        @ ops.mainfunc
        def forward(self, x, state):

            x = self.layernorm(
                self.processEmbed(self.getIndex(self.emb, x)),
                self.emb1, self.emb2)

            statea = state[0::4]
            stateb = state[1::4]
            statec = state[2::4]
            stated = state[3::4]

            ot = []

            for i in range(self.n_layers):
                x, aaa, bbb, ccc, ddd = self.doLayer(
                    x, statea[i], stateb[i], statec[i], stated[i], i)

                ot = ot + [aaa, bbb, ccc, ddd]

            x = self.matvec(self.postprocess2, self.layernorm(x, self.postprocess0,
                                                              self.postprocess1))

            return self.postProcessTensor(x[-1]), self.stack(ot)

        # for keras stuff, ignore this if you are not using keras
        def call(self, *args, **kwds):
            del kwds["training"]
            return self.forward(*args, **kwds)
    returnObject: myRWKV = ops.postProcessModule(myRWKV(*args))
    return returnObject
