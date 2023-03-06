import numpy as np

import rwkvstic.agnostic.backends.base as RWKVOp


class RWKVNumpyOps(RWKVOp.module):
    def __init__(self, layers, embed, *args, **kwargs):
        super().__init__(layers, embed, *args, **kwargs)

        self.initTensor = lambda x: x.float().cpu().numpy()
        self.sqrt = np.sqrt
        self.mean = np.mean
        self.relu = lambda x: np.maximum(x, 0)
        self.exp = np.exp
        self.stack = lambda x: x
        self.matvec = lambda x, y: np.matmul(x, y.T).T
        self.prod = lambda x: np.prod(x, axis=1)
        self.lerp = lambda x, y, z: x*(1-z) + y*(z)
        self.minimum = lambda x, y: np.minimum(x, y)
        self.maximum = lambda x, y: np.maximum(x, y)

        self.klimit = [18] * embed
        # module def
        self.module = object
        self.log = np.log

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x
        def getIndex(x, y): return [x[z] for z in y]
        self.getIndex = getIndex

        def ln(x, w, b):
            xee2 = x - self.mean(x, axis=1,
                                 keepdims=True)

            x2 = self.sqrt(self.mean(xee2*xee2, axis=1,
                           keepdims=True) + 0.000009999999747378752)

            return w*(xee2/x2) + b
        self.layernorm = ln


class RWKVJaxOps(RWKVOp.module):
    def __init__(self, layers, embed, *args, dtype=None, preJax=False, **kwargs):
        from jax import numpy as npjax
        super().__init__(layers, embed, *args, **kwargs)
        import inquirer
        self.dtype = dtype if dtype else inquirer.prompt([inquirer.List(
            'dtype', message="Choose a dtype", choices=[npjax.bfloat16, npjax.float16, npjax.float32, npjax.float64])])['dtype']
        if preJax:
            self.initTensor = lambda x: npjax.array(x, dtype=dtype)
        else:
            self.initTensor = lambda x: npjax.array(
                x.float().cpu().numpy(), dtype=dtype)
        self.sqrt = lambda x: npjax.sqrt(x)
        self.mean = npjax.mean
        self.relu = lambda x: npjax.maximum(x, 0)
        self.exp = lambda x: npjax.exp(x)
        self.stack = lambda x: npjax.array(x, dtype=dtype)
        self.mnstack = lambda x: npjax.array(x, dtype=dtype)
        self.matvec = lambda x, y: npjax.matmul(x, y.T).T
        self.prod = lambda x: npjax.prod(x, axis=1)
        self.lerp = lambda x, y, z: npjax.array(x)*(1-z) + npjax.array(y)*(z)
        self.minimum = lambda x, y: npjax.minimum(x, y)
        self.maximum = lambda x, y: npjax.maximum(x, y)
        # module def
        self.module = object
        self.log = npjax.log

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x
        def getIndex(x, y): return npjax.array([x[z] for z in y])

        def scatter(x, y, z):
            x = x.at[y].set(z)
            return x

        self.scatterindices = [slice(2, 5), slice(2, 4), slice(0, 2)]

        self.scatter = scatter
        self.getIndex = getIndex
        # in pgetInostfunc, convert to numpy

        def ln(x, w, b):
            xee2 = x - self.mean(x, axis=1, keepdims=True)

            x2 = self.sqrt(self.mean(xee2*xee2, axis=1,
                           keepdims=True) + 0.000009999999747378752)

            return w*(xee2/x2) + b

        self.layernorm = ln
        self.emptyState = npjax.array(self.emptyState)
