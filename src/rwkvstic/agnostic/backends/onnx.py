import numpy as np

import rwkvstic.agnostic.backends.base as RWKVOp


class RWKVOnnxOps(RWKVOp.module):

    def __init__(self, layers, embed, *args, **kwargs):
        import onnx

        super().__init__(layers, embed, *args, **kwargs)
        print("embed ", embed)

        dtype = onnx.TensorProto.FLOAT

        self.nm = 0

        self.TensorList = []
        self.NodeList = []

        def initTensor(x):
            name = f"PreTrainedTensor_{self.nm}"
            self.nm += 1
            if isinstance(x, list):
                xx = np.array(x)
            else:
                xx = x.squeeze().float().cpu().numpy()
                # convert to float32
                xx = xx.astype(np.float32)
            rrx = (name, xx)

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
                outputs=[name]
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

        def matvec(x, y):
            name = f"matvec_{self.nm}_out"
            oname = f"matvec_g_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'MatMul',
                inputs=[x, y],
                outputs=[name]
            )
            self.NodeList.append(node)
            return name

        self.matvec = matvec

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
            return self.add(self.multiply(y, z), self.multiply(x, self.subtract(self.one, z)))

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
            name = f"layernorm_{self.nm}_out"
            self.nm += 1
            node = onnx.helper.make_node(
                'LayerNormalization',
                inputs=[x, w, b],
                outputs=[name]
            )
            self.NodeList.append(node)

            return name

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

            return squeeze(name)

        self.stackEmbed = False

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

        # convert to float32
        self.emptyState = np.array(self.emptyState, dtype=np.float32)

        # self.zero = initTensor([0.0]*embed)

        def ppm(x):
            inputtensor = onnx.helper.make_tensor_value_info("input0",
                                                             onnx.TensorProto.INT32,
                                                             [1]), "input0"

            emptyState = list(map(lambda x: (onnx.helper.make_tensor_value_info("instate"+str(x),
                                                                                dtype,
                                                                                [embed]), "instate"+str(x)), range(5*layers)))
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
                        list(map(lambda x:x[0], emptyState)), *[onnx.helper.make_tensor_value_info(x[0],
                                                                                                   dtype,
                                                                                                   x[1].shape) for x in self.TensorList]],

                outputs=[logits, *state],  # Graph output


                # did not work, needs to be external

                # external_initializers=list(
                #     map(lambda x: x[0], self.TensorList))


            )

            modelDef = onnx.helper.make_model(
                graph_def, producer_name="rwkvstic",
            )

            # onnx.external_data_helper.convert_model_to_external_data(
            #     modelDef, location="onnx")
            # # make all initializers external

            # onnx.external_data_helper.write_external_data_tensors(
            #     modelDef, "onnx")

            modelDef.opset_import[0].version = 17

            model_def = onnx.shape_inference.infer_shapes(modelDef)

            # onnx.checker.check_model(model_def)

            # onnx.save(proto=model_def,
            #           save_as_external_data=True, f="model.onnx", location="model.bin", size_threshold=0, all_tensors_to_one_file=True, convert_attribute=True)

            # make temp dir
            # import os
            # try:
            #     os.mkdir("onnx")
            # except:
            #     pass
            if (False):
                # save all tensors with onnx
                for i in self.TensorList:
                    print(i)
                    onnx.save_tensor(i, f"onnx/{i.name}.onnx")

                # zip all tensors
                import zipfile
                with zipfile.ZipFile('onnx.zip', 'w') as zipObj:
                    # Iterate over all the files in directory
                    for folderName, subfolders, filenames in os.walk('onnx'):
                        for filename in filenames:
                            # create complete filepath of file in directory
                            filePath = os.path.join(folderName, filename)
                            # Add file to zip
                            zipObj.write(filePath, filename)

                    # add model
                    zipObj.write("model.onnx", "model.onnx")

                # delete temp dir
                import shutil
                shutil.rmtree("onnx")

            # run model
            import onnxruntime as rt

            # session execution provider options
            sess_options = rt.SessionOptions()
            # for x in self.TensorList:
            #     sess_options.add_initializer(x[0], rt.OrtValue.ortvalue_from_numpy(
            #         x[1]))
            # create session providers
            # print all providers
            print(rt.get_available_providers())
            providers = ["CPUExecutionProvider"]

            sess = rt.InferenceSession(
                model_def.SerializeToString(), sess_options, providers=providers)

            ins = {

            }
            for i in range(self.TensorList.__len__()):
                ins[self.TensorList[i][0]
                    ] = self.TensorList[i][1].astype(np.float32)

            class interOp():
                def forward(selff, xi, statei):
                    # print(statei[0][23])
                    # create inputs
                    inputs = ins
                    # get input names
                    input_names = [inputtensor[1], *
                                   list(map(lambda x: x[1], emptyState)), *[x[0] for x in self.TensorList]]
                    # get output names
                    output_names = [outs[0], *outs[1]]
                    # print(output_names)

                    # create input dict
                    inputs[input_names[0]] = np.array([xi[-1]], dtype=np.int32)
                    for i in range(5*layers):
                        inputs[input_names[i+1]] = statei[i]

                    outputs = sess.run(output_names, inputs)
                    # print(outputs[1][23])

                    return outputs[0], outputs[1:]

            return interOp()
        self.postProcessModule = ppm

# In the code above, the error was in line
