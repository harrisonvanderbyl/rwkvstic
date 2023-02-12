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
        self.add = lambda x, y: x+y
        self.divide = lambda x, y: x/y
        self.multiply = lambda x, y: x*y
        self.subtract = lambda x, y: x-y
        self.stack = raiseNotImplemented
        self.matvec = raiseNotImplemented
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
        self.initfunc = raiseNotImplemented
        self.layerdef = raiseNotImplemented
        self.mainfunc = raiseNotImplemented
        self.emptyState: self.MatrixType = []

        def logistical(x: self.VectorType) -> self.VectorType:
            return 1 / (self.exp(x) + 1)
        self.logistical = logistical
        self.postProcessModule = lambda x: x

        self.sample = npsample

        # typing, set as any
        self.tensorDef = None
