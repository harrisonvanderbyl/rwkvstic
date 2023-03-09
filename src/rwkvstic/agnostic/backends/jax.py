

import rwkvstic.agnostic.backends.base as RWKVOp


class RWKVNumpyOps(RWKVOp.module):
    
    def __init__(self, layers, embed, *args, **kwargs):
        import numpy as np
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

class RWKVCuPyOps(RWKVOp.module):
    def __init__(self, layers, embed, *args, **kwargs):
        import cupy as np
        from cupyx import jit
        super().__init__(layers, embed, *args, **kwargs)
        with np.cuda.Device(0):
            np.cuda.device.get_cublas_handle()
            import numpy
            dtype = numpy.float32
            mattype = numpy.float32
            self.initTensor = lambda x: np.array(x.float().cpu().numpy(),dtype=dtype) if len(x.squeeze().shape) ==1 else np.array(x.float().cpu().numpy(),dtype=mattype).T
            self.initCpuTensor = lambda x: np.array(x.float().cpu().numpy(),dtype=dtype)
            self.sqrt = np.sqrt
            self.mean = np.mean
            self.relu = lambda x: np.maximum(x, 0)
            self.exp = np.exp
            self.stack = np.stack
            self.matvec = lambda x, y: np.matmul(y,x)
            self.prod = lambda x: np.prod(x, axis=1)
            self.lerp = lambda x, y, z: x*(1-z) + y*(z)
            self.minimum = lambda x, y: np.minimum(x, y)
            self.maximum = lambda x, y: np.maximum(x, y)
            self.roll = lambda x: np.roll(x,1,axis=0)
            self.emptyState = np.array(self.emptyState,dtype=dtype)
            self.klimit = [18] * embed
            # module def
            self.module = object
            self.log = np.log
            self.mnstack = lambda x: x
            # pytorch function defs
            self.initfunc = lambda x: x
            self.layerdef = lambda x: x
            self.mainfunc = lambda x: x
            def getIndex(x, y): return np.stack([x[z] for z in y])
            self.stackEmb =  True
            self.getIndex = getIndex
            self.scatterindices = [slice(2, 5), slice(2, 4), slice(0, 2)]
            self.postProcessTensor = lambda x: x.get()
            
            def ln(x, w, b):
                xee2 = x - np.mean(x, axis=1,
                                    keepdims=True)

                x2 = np.sqrt(np.mean(np.square(xee2), axis=1,
                            keepdims=True) + 0.000009999999747378752)

                return w*(xee2/x2) + b
            self.layernorm = ln
            def ppm(x):
                
                mempool = np.get_default_memory_pool()
                pinned_mempool = np.get_default_pinned_memory_pool()

               
                # You can access statistics of these memory pools.
                print(mempool.used_bytes()/1024/1024/1024)              # 0
                print(mempool.total_bytes()/1024/1024/1024)             # 0
                print(pinned_mempool.n_free_blocks())    # 0
                return x
            self.postProcessModule = ppm

class RWKVCuPyQuantOps(RWKVCuPyOps):
    def __init__(self, layers, embed, *args, **kwargs):
        super().__init__(layers, embed, *args, **kwargs)
        import cupy as np

        def QuantizeMatrix(x):
            rang = 255
            ran, mini = (x.max(0)[0]-x.min(0)[0])/rang,  x.min(0)[0]
            x = x.double()
            x = ((x-mini)/ran)
            
            x = np.array(x.float(), dtype=np.uint8)

            return [x, np.array(ran.float()), np.array(mini.float())]

        def QuantizedMatVec(x, y):
            if len(x) != 3:
                return y @ x
            rx, spread, zpoint = x
            yy = y*spread

            xmain = yy@rx
            zp = (y@zpoint).reshape(-1, 1)
            return xmain + zp

        def initTensor(x):
            if len(x.squeeze().shape) == 1:
                return np.array(x.float().cpu().numpy(), dtype=np.float32)
            
            splitmatrices = x.chunk(32, 1)

            xx = [QuantizeMatrix(x)
                  for x in splitmatrices]
            xxo = np.concatenate([x[0] for x in xx], 1).T
            xx1 = np.concatenate([x[1] for x in xx])
            xx2 = np.concatenate([x[2] for x in xx])
            return xxo, xx1, xx2    

        self.initTensor = initTensor
        self.matvec = QuantizedMatVec

                

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
