from rwkvstic.agnostic.backends.base import module


class RWKVCoreMLOps(module):
    def __init__(self, layers, embed, *args, **kwargs):

        super().__init__(layers, embed, *args, **kwargs)
        from coremltools.converters.mil.mil import Builder as mb
        from coremltools.converters.mil.mil import Program, Function
        func_inputs = {"x": mb.placeholder(shape=[1]),
                       **{f"state{i}": mb.placeholder(shape=[1]) for i in range(5*layers)},
                       }

        with Function(func_inputs) as ssa_fun:

            prog = Program()

            self.initTensor = lambda x: x.float().cpu().numpy()
            self.sqrt = mb.sqrt
            self.mean = mb.reduce_mean
            self.relu = lambda x: mb.maximum(x, 0)
            self.exp = mb.exp
            self.stack = lambda x: x
            self.matvec = mb.matmul
            self.prod = lambda x: mb.prod(x, axis=1)
            self.lerp = lambda x, y, z: x*(1-z) + y*(z)
            self.minimum = lambda x, y: mb.minimum(x, y)
            self.maximum = lambda x, y: mb.maximum(x, y)
            self.module = object

            self.log = mb.log

            self.getIndex = lambda x, y: mb.gather((x, y))

            def ln(x, w, b):
                xee2 = x - self.mean(x)

                x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

                return w*(xee2/x2) + b
            self.layernorm = ln

            def ppm(x):
                inp0, inp1 = ssa_fun.inputs["x"], [
                    ssa_fun.inputs[f"state{i}"] for i in range(5*layers)]
                re = x.forward(inp0, inp1)
                print(re)
                ssa_fun.set_outputs([re])

        self.postProcessModule = ppm
