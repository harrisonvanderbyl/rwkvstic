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
        self.RnnOnly = False
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
        self.intTensor = lambda x: [x] if type(x) == int else x
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
        self.emptyState: self.MatrixType = np.array([(([[0.00]*embed, [0.00]*embed, [0.00]*embed, [
            0.00]*embed]+([[-1e30]*embed] if self.useLogFix else [])))]*layers)

        print(self.emptyState.shape)

        def logistical(x: self.VectorType) -> self.VectorType:
            return 1 / (self.exp(-x) + 1)

        self.logistical = logistical
        self.neg = lambda x: -x
        self.postProcessModule = lambda x: x
        self.mnstack = lambda x: x

        self.sample = npsample
        def emptyarray(x: int): return [0]*x
        self.emptyarray = emptyarray
        def arrayPush(x: list, y, i: int): return [*x[0:i], y, *x[i+1:]]
        self.arrayPush = arrayPush
        def rng(x: int): return range(x)
        self.rng = rng
        def pop(x): return x[-1]
        self.pop = pop
        def arrayGet(x, i: int): return x[i]
        self.arrayGet = arrayGet

        def push(x, y):
            x[0] = y
            return x
        self.push = push

        self.roll = lambda x: [*x[0:1], *x[0:-1]]
        # typing, set as any

        self.tensorDef = None

        def lenn(x): return len(x)
        self.len = lenn

        self.stackEmb = False

        def getIndex(x, y): return x[y]

        self.getIndex = getIndex

        def scatter(x, y, z):
            x[y] = z
            return x
        self.scatterindices = [slice(2, 5), slice(2, 4), slice(0, 2)]

        self.scatter = scatter


class rnnmodule:
    def __init__(self, layers, embed, *args, useLogFix=True, **kwargs):

        from rwkvstic.agnostic.samplers.numpy import npsample
        self.RnnOnly = True
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
        self.initfunc = raiseNotImplemented
        self.layerdef = raiseNotImplemented
        self.mainfunc = raiseNotImplemented
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

        # typing, set as any

        self.tensorDef = None

        self.stackEmb = False

        self.getIndex = lambda x, y: x[y[-1]]
