from rwkvstic.rwkvMaster import RWKVMaster
from rwkvstic.agnostic.samplers.numpy import npsample


def initTFLiteFile(path, tokenizer=None):
    import tensorflow.lite as tflite

    import tensorflow as tf

    interpreter = tflite.Interpreter(
        model_path=path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    embed = input_details[1]['shape'][1]
    layers = input_details[1]['shape'][0]
    dtype = input_details[1]['dtype']

    class InterOp():
        def forward(self, x, y):

            interpreter.set_tensor(
                input_details[0]['index'], tf.convert_to_tensor(x, dtype=tf.int32))
            interpreter.set_tensor(
                input_details[1]['index'], y)
            interpreter.invoke()
            output_data = interpreter.get_tensor(
                output_details[0]['index']), interpreter.get_tensor(output_details[1]['index'])

            return output_data
    model = InterOp()
    emptyState = tf.convert_to_tensor(
        [[0.01]*int(embed)]*int(layers), dtype=dtype)

    def initTensor(x): return tf.convert_to_tensor(x, dtype=dtype)
    return RWKVMaster(model, emptyState, initTensor, npsample, tokenizer)
