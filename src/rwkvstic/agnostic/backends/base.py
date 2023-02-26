from typing import List, Dict, Union


class module:
    def __init__(self, layers, embed, *args, useLogFix=True, **kwargs):

        from rwkvstic.agnostic.samplers.numpy import npsample
        self.VectorType = List[float]
        self.useLogFix = useLogFix
        self.MatrixType = List[List[float]]
        self.TensorType = Union[self.VectorType, self.MatrixType]
        self.MatVec = (
            self.MatrixType, self.VectorType), self.VectorType

        def raiseNotImplemented(*args: List[self.TensorType], **kwargs: Dict[str, self.TensorType]) -> self.TensorType:
            print(NotImplementedError())
            raise
        print("init RWKVOPS, from super")
        self.initTensor = raiseNotImplemented
        self.n_layers = layers
        self.processEmbed = lambda x: x
        self.initCpuTensor = lambda x: self.initTensor(x)
        self.sqrt = raiseNotImplemented
        self.mean = raiseNotImplemented
        self.relu = raiseNotImplemented
        self.exp = raiseNotImplemented
        self.maximum = raiseNotImplemented
        def add(x, y): return x+y
        def divide(x, y): return x/y
        def multiply(x, y): return x*y
        def subtract(x, y): return x-y
        self.intTensor = lambda x: x
        self.add = add
        self.divide = divide
        self.multiply = multiply
        self.subtract = subtract
        self.stack = raiseNotImplemented
        self.matvec = raiseNotImplemented
        self.prod = raiseNotImplemented
        self.layernorm = raiseNotImplemented

        self.lerp = raiseNotImplemented

        def ppt(x: self.VectorType):
            return x
        self.postProcessTensor = ppt
        # module def
        self.module = raiseNotImplemented
        self.log = raiseNotImplemented
        self.minimum = raiseNotImplemented
        self.klimit = raiseNotImplemented
        # tensorflow function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x

        self.mainfunc = lambda x: x

        import numpy as np
        self.emptyState: self.MatrixType = np.array((([[0.00]*embed, [0.00]*embed, [0.00]*embed, [
            0.00]*embed]+([[-1e30]*embed] if self.useLogFix else [])))*layers)

        print(self.emptyState.shape)

        def logistical(x: self.VectorType) -> self.VectorType:
            return 1 / (self.exp(-x) + 1)

        self.logistical = logistical
        self.neg = lambda x: -x
        self.postProcessModule = lambda x: x

        self.sample = npsample

        self.roll = raiseNotImplemented
        # typing, set as any

        self.tensorDef = None

        def len(x): return len(x)
        self.len = len

        self.stackEmb = False

        def getIndex(x, y): return x[y]
        self.getIndex = getIndex
