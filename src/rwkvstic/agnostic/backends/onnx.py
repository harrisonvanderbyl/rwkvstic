import numpy as np

import rwkvstic.agnostic.backends.base as RWKVOp


class RWKVOnnxOps(RWKVOp.module):

    def __init__(self, layers, embed, *args, dtype=None, **kwargs):
        import onnx

        super().__init__(layers, embed, *args, **kwargs)
        print("embed ", embed)
        import torch
        if dtype is None:
            import inquirer

            questions = [
                inquirer.List('dtype',
                              message="What dtype do you want to use?",
                              choices=[torch.float32,
                                       torch.float16],

                              ),
            ]
            dtype = inquirer.prompt(questions)["dtype"]

        dtype = onnx.TensorProto.FLOAT if dtype == np.float32 else onnx.TensorProto.FLOAT16 if dtype == torch.float16 else onnx.TensorProto.BFLOAT16 if dtype == torch.bfloat16 else onnx.TensorProto.FLOAT
        nptype = np.float32 if dtype == onnx.TensorProto.FLOAT else np.float16 if dtype == onnx.TensorProto.FLOAT16 else np.float16 if dtype == onnx.TensorProto.BFLOAT16 else np.float32

        self.nm = 0
        exportname = f"RWKV_{layers}_{embed}.onnx"
        externalname = f"RWKV_{layers}_{embed}.bin"

        # remove old files
        import os
        if os.path.exists(exportname):
            os.remove(exportname)
        if os.path.exists(externalname):
            os.remove(externalname)

        self.TensorList = []
        self.NodeList = []

        def initTensor(x, dtype=dtype):
            name = f"PreTrainedTensor_{self.nm}"
            self.nm += 1
            if isinstance(x, list):
                xx = np.array(x).astype(nptype)
            elif isinstance(x, np.ndarray):
                xx = x
            else:
                xx = x.squeeze().float().cpu().numpy()
                # convert to float32
                xx = xx.astype(nptype)
            rrx = onnx.helper.make_tensor(
                name,
                dtype,
                xx.shape,
                xx.tobytes(),
                raw=True

            )

            onnx.external_data_helper.set_external_data(
                rrx,
                location=externalname,

            )

            self.TensorList.append(rrx)
            return name

        self.initTensor = initTensor

        def sqrt(x):
            name = f"sqrt_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Sqrt',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.sqrt = sqrt

        def mean(x):
            name = f"mean_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'ReduceMean',
                inputs=[x],
                outputs=[name],
                axes=[1],
                keepdims=1
            )
            self.NodeList.append(node)

            return name

        self.mean = mean

        def relu(x):
            name = f"relu_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Relu',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.relu = relu

        def exp(x):
            name = f"exp_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Exp',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.exp = exp

        def stack(x):
            return x

        self.stack = stack

        def transpose(x):
            name = f"transpose_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Transpose',
                inputs=[x],
                outputs=[name],
                perm=[1, 0]
            )
            self.NodeList.append(node)

            return name

        def matvec(x, y):
            name = f"matvec_{self.nm}_out"
            oname = f"matvec_g_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'MatMul',
                inputs=[x, transpose(y)],
                outputs=[name],

            )
            self.NodeList.append(node)
            return transpose(name)

        self.matvec = matvec

        def prod(x):
            name = f"prod_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'ReduceProd',
                inputs=[x],
                outputs=[name],
                axes=[1],
                keepdims=0


            )
            self.NodeList.append(node)

            return name

        self.prod = prod

        def mul(x, y):
            name = f"mul_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Mul',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.multiply = mul

        def squeeze(x):
            name = f"squeeze_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Squeeze',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        def add(x, y):

            name = f"add_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Add',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.add = add

        def sub(x, y):
            name = f"sub_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Sub',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.subtract = sub

        self.one = initTensor([1.0]*embed)

        def lerpx(x, y, z):
            return self.add(x, self.multiply(self.subtract(y, x), z))

        self.lerp = lerpx

        def minimum(x, y):
            name = f"minimum_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Min',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name
        self.minimum = minimum
        # module def
        self.module = object

        def log(x):
            name = f"log_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Log',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.log = log

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x

        def divide(x, y):
            name = f"divide_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Div',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.divide = divide

        def layernorm(x, w, b):

            # layernorm batchnorm
            xee2 = self.subtract(x, self.mean(x))

            x2 = self.sqrt(self.mean(self.multiply(
                xee2, xee2)))
            o = self. add(
                self.multiply(w,
                              self.divide(xee2, x2)), b)

            return o
            # nan on cuda
            # return torch.nn.functional.layer_norm(x, x.shape, w.expand(x.shape), b.expand(x.shape))

        self.layernorm = layernorm

        def getIndex(x, y):
            name = f"getIndex_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Gather',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.stackEmbed = False

        def pop(x):
            name = f"pop_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Gather',
                inputs=[x, self.subtract(self.len(x), self.one)],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.pop = pop

        def neg(x):
            name = f"neg_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Neg',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.neg = neg

        def logistic(x):
            name = f"logistic_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Sigmoid',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name
        self.logistical = logistic

        def maximum(x, y):
            name = f"maximum_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Max',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.maximum = maximum

        self.getIndex = getIndex

        self.zero = initTensor(
            np.zeros([1], dtype=np.int64), dtype=onnx.TensorProto.INT64)
        self.one = initTensor(
            np.ones([1], dtype=np.int64), dtype=onnx.TensorProto.INT64)

        # convert to float32
        self.emptyState = np.array(self.emptyState, dtype=nptype)

        def lenn(x):
            name = f"lenn_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Shape',
                inputs=[x],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.len = lenn

        def roll(x):
            name = f"roll_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Roll',
                inputs=[x, self.one],
                outputs=[name],
                axis=0
            )
            self.NodeList.append(node)

            return name

        self.roll = roll

        def push(x, y, z):
            # set index 0 to y
            name = f"push_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'Scatter',
                inputs=[x, z, y],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

        self.arrayPush = push

        def Push(x, y):
            return self.arrayPush(x, y, self.zero)

        self.push = Push
        # self.zero = initTensor([0.0]*embed)
        emb = initTensor([[0.0]*embed])

        def emptyTensor(x):
            name = f"emptyTensor_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                "Expand",
                inputs=[emb],
                outputs=[name],
                shape=[x]
            )

            self.NodeList.append(node)

            return name

        self.emptyarray = emptyTensor

        self.arrayGet = getIndex

        def emptyTensor(x):
            return []
        self.mainarray = emptyTensor
        self.pushstate = lambda x, y, z: x + [y]

        def loop(x: int, ww, xx: int, k, v, bz, cz, gz, rz):
            print("loop")
            print(x)
            print(ww)
            print(xx)
            print(k)
            print(v)
            print(bz)
            print(cz)
            print(gz)
            print(rz)
            return bz, cz, gz, rz

        self.loop = loop

        def ppm(x):
            inputtensor = onnx.helper.make_tensor_value_info("input0",
                                                             onnx.TensorProto.INT32,
                                                             [None]), "input0"

            emptyState = list(map(lambda x: (onnx.helper.make_tensor_value_info("instate"+str(x),
                                                                                dtype,
                                                                                [embed]), "instate"+str(x)), range((4+self.useLogFix)*layers)))
            outs = x.forward(
                inputtensor[1], list(map(lambda x: x[1], emptyState)))
            print(self.TensorList.__len__())
            print(self.NodeList.__len__())
            print(outs)
            logits = onnx.helper.make_tensor_value_info(outs[0],
                                                        dtype,
                                                        [50277])
            state = list(map(lambda x: onnx.helper.make_tensor_value_info(x,
                                                                          dtype,
                                                                          [embed]), outs[1]))

            # Create the graph (GraphProto)
            graph_def = onnx.helper.make_graph(
                nodes=self.NodeList,  # The list of nodes in the graph.
                name="RWKV",
                # Graph input

                inputs=[inputtensor[0], * \
                        list(map(lambda x:x[0], emptyState))],

                outputs=[logits, *state],  # Graph output

                initializer=self.TensorList,  # initializer



                # did not work, needs to be external

            )

            modelDef = onnx.helper.make_model(
                graph_def, producer_name="rwkvstic",


            )

            modelDef.opset_import[0].version = 17

            onnx.save(modelDef, exportname)

            # run model
            print("Model saved to: ", exportname, " and is ready to be run")
            print("Data type: ", dtype)
            print("Embedding size: ", embed)
            print("Number of layers: ", layers)
            print("external data: ", externalname)
            exit()
        self.postProcessModule = ppm
