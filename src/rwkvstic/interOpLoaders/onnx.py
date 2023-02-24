
from rwkvstic.agnostic.samplers.numpy import npsample
from rwkvstic.rwkvMaster import RWKVMaster


def initONNXFile(path, tokenizer=None):
    import onnxruntime as rt

    # session execution provider options
    sess_options = rt.SessionOptions()

    print(rt.get_available_providers())
    import inquirer
    providers = inquirer.checkbox(
        "Select execution providers", choices=rt.get_available_providers())
    print(providers)

    sess = rt.InferenceSession(
        path, sess_options, providers=providers)

    ins = {

    }

    embed = int(path.split("_")[2].split(".")[0])
    layers = int(path.split("_")[1])
    typenum = sess.get_inputs()[1].type
    print(typenum)
    import numpy as np

    if typenum == "tensor(float)":
        typenum = np.float32
    elif typenum == "tensor(float16)":
        typenum = np.float16
    elif typenum == "tensor(bfloat16)":
        typenum = np.bfloat16

    class InterOp():
        def forward(selff, xi, statei):
            # print(statei[0][23])
            # create inputs
            inputs = ins
            # get input names
            input_names = sess.get_inputs()
            input_names = [x.name for x in input_names]
            # get output names
            output_names = sess.get_outputs()
            output_names = [x.name for x in output_names]
            # print(output_names)

            # create input dict
            inputs[input_names[0]] = np.array([xi[-1]], dtype=np.int32)
            for i in range(len(input_names)-1):
                inputs[input_names[i+1]] = statei[i]

            outputs = sess.run(output_names, inputs)
            # print(outputs[1][23])

            return outputs[0], outputs[1:]
    model = InterOp()

    # emptyState = []
    emptyState = [
        np.array([0.01]*int(embed)).astype(typenum)]*len(sess.get_inputs()[1:])

    def initTensor(x): return np.array(x, dtype=typenum)
    return RWKVMaster(model, emptyState, initTensor, npsample, tokenizer)
