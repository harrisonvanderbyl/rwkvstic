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
        self.mainarray = self.emptyarray
        self.pushstate = self.arrayPush


class RWKVJaxOps(RWKVOp.module):
    def __init__(self, layers, embed, *args, preJax=False, **kwargs):
        from jax import numpy as npjax
        super().__init__(layers, embed, *args, **kwargs)
        if preJax:
            self.initTensor = lambda x: npjax.array(x)
        else:
            self.initTensor = lambda x: npjax.array(x.float().cpu().numpy())
        self.sqrt = lambda x: npjax.sqrt(x)
        self.mean = npjax.mean
        self.relu = lambda x: npjax.maximum(x, 0)
        self.exp = lambda x: npjax.exp(x)
        self.stack = lambda x: npjax.array(x)
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
        self.getIndex = getIndex
        # in postfunc, convert to numpy

        def ln(x, w, b):
            xee2 = x - self.mean(x, axis=1, keepdims=True)

            x2 = self.sqrt(self.mean(xee2*xee2, axis=1,
                           keepdims=True) + 0.000009999999747378752)

            return w*(xee2/x2) + b

        self.layernorm = ln
        self.emptyState = npjax.array(self.emptyState)
