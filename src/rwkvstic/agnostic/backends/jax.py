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
        self.matvec = np.matmul
        self.lerp = lambda x, y, z: x*(1-z) + y*(z)
        self.minimum = lambda x, y: np.minimum(x, y)
        self.minimum = lambda x, y: np.maximum(x, y)
        self.klimit = [18] * embed
        # module def
        self.module = object
        self.log = np.log

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x

        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b
        self.layernorm = ln
        self.emptyState = [[0.01]*embed]*(4+self.useLogFix)*layers


class RWKVJaxOps(RWKVOp.module):
    def __init__(self, layers, embed, *args, preJax=False, **kwargs):
        from jax import numpy as npjax
        super().__init__(layers, embed, *args, **kwargs)
        if preJax:
            self.initTensor = lambda x: npjax.array(x)
        else:
            self.initTensor = lambda x: npjax.array(x.float().cpu().numpy())
        self.sqrt = lambda x: npjax.sqrt(x)
        self.mean = lambda x: npjax.mean(x)
        self.relu = lambda x: npjax.maximum(x, 0)
        self.exp = lambda x: npjax.exp(x)
        self.stack = lambda x: x
        self.matvec = npjax.matmul
        self.lerp = lambda x, y, z: x*(1-z) + y*(z)
        self.minimum = lambda x, y: npjax.minimum(x, y)
        self.maximum = lambda x, y: npjax.maximum(x, y)
        self.klimit = npjax.array([18] * embed)
        # module def
        self.module = object
        self.log = npjax.log

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x
        # in postfunc, convert to numpy

        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b

        self.layernorm = ln
        self.emptyState = npjax.array([[0.01]*embed]*(4+self.useLogFix)*layers)
