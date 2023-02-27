import inquirer
import os
import rwkvstic.agnostic.backends.base as RWKVOp


class RWKVTFOps(RWKVOp.module):
    def __init__(self, layers, embed, *args, useGPU: bool = None, **kwargs):
        try:
            import tensorflow as tf
        except:
            inst = inquirer.confirm(
                "Tensorflow not installed, do you want to install it?")
            if inst:
                os.system("pip3 install tensorflow")
                import tensorflow as tf
        if (not (inquirer.confirm("Do you want to use GPU?") if useGPU is None else useGPU)):
            tf.config.experimental.set_visible_devices([], "GPU")
        tf.config.optimizer.set_jit(True)
        tf.config.optimizer.set_experimental_options(
            {"auto_mixed_precision": True})

        super(RWKVTFOps, self).__init__(layers, embed, *args, **kwargs)
        self.initTensor = lambda x: tf.convert_to_tensor(
            x.float().cpu().numpy())

        self.sqrt = tf.sqrt
        self.mean = tf.reduce_mean
        self.relu = lambda x: tf.maximum(x, tf.zeros_like(x))
        self.minimum = tf.minimum
        self.maximum = tf.maximum
        self.exp = tf.exp
        self.unsqueeze = tf.expand_dims

        self.cat = lambda x: tf.concat(x, axis=0)

        def matvec(x, y):
            y = tf.transpose(y)
            return tf.transpose(tf.matmul(x, y))
        self.matvec = matvec
        self.prod = lambda x: tf.reduce_prod(x, axis=1)

        def roll(x):
            zx = x
            rx = tf.concat([zx[:1], zx[0:-1]], axis=0)
            return rx
        self.rng = lambda x: tf.range(x, dtype=tf.int32)

        def emptyarray(x): return tf.TensorArray(tf.float32, size=x, element_shape=(
            tf.TensorShape([embed])
        ))
        self.emptyarray = emptyarray
        def arrayPush(x, y, i): return x.write(i, y)
        self.arrayPush = arrayPush
        def arrayGet(x, i): return x.read(i)
        self.arrayGet = arrayGet

        def stack(x):
            return tf.stack(x) if type(x) == list else x.stack()
        self.stack = stack
        self.roll = roll
        def pop(x): return x[-1]
        self.pop = pop

        def push(x, y):
            # x[0] = y
            # tensorflow does not support item assignment
            # so we have to do this
            return tf.concat([tf.expand_dims(y, 0), x[1:]], axis=0)

        self.push = push
        self.log = tf.math.log
        self.lerp = lambda x, y, z: x*(1-z)+y*z
       # module def
        self.module = tf.Module

        self.divide = tf.math.truediv
        # class def
       # tensorflow function defs
        self.initfunc = lambda x: x
        self.layerdef = tf.function(
            input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)]+(4+self.useLogFix)*[tf.TensorSpec(shape=[None], dtype=tf.float32)]+[tf.TensorSpec(dtype=tf.int32, shape=None)])

        self.mainfunc = tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int64), tf.TensorSpec(
            shape=[(4+self.useLogFix)*layers, embed], dtype=tf.float32)])
        self.emptyState = tf.convert_to_tensor(
            self.emptyState, dtype=tf.float32)

        def ln(x, w, b):
            # layernorm batchnorm
            xee2 = x - tf.reduce_mean(x, 1, True)

            x2 = tf.sqrt(tf.reduce_mean(xee2*xee2, 1, True) +
                         1e-5)
            o = w*(xee2/x2) + b
            return o

        self.layernorm = ln

        self.len = lambda x: tf.shape(x)[0]
        self.getIndex = lambda x, y: tf.gather(x, y, axis=0)
        self.mainarray = self.emptyarray
        self.pushstate = self.arrayPush


class RWKVTFExport(RWKVTFOps):
    def __init__(self, layers, embed, *args,  exports=None, **kwargs):
        super(RWKVTFExport, self).__init__(layers, embed, *args, **kwargs)
        import tensorflow as tf
        self.module = tf.keras.Model
        path = f"tfdist/rwkv-{layers}-{embed}/"

        def save(x):
            x([0], self.emptyState)
            try:
                try:
                    os.mkdir("tfdist")
                except:
                    pass
                os.mkdir(path)
            except:
                pass

            q = exports if exports is not None else inquirer.checkbox(message="What to export?", choices=[
                "savedmodel32", "tflite32", "tflite16"])

            if "savedmodel32" in q:
                try:
                    os.mkdir(path+"sm")
                except:
                    pass

                tf.keras.models.save_model(x, path+"sm/whole")

            if "tflite32" in q:
                try:
                    os.mkdir(path+"tflite32")
                except:
                    pass

                converter = tf.lite.TFLiteConverter.from_concrete_functions(
                    [x.forward.get_concrete_function()])
                tflite_model = converter.convert()
                open(f"model-{layers}-{embed}-32.tflite",
                     "wb").write(tflite_model)

            if "tflite16" in q:
                try:
                    os.mkdir(path+"tflite16")
                except:
                    pass

                converter = tf.lite.TFLiteConverter.from_concrete_functions(
                    [x.forward.get_concrete_function()])
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                tflite_model = converter.convert()
                open(f"model-{layers}-{embed}-16.tflite",
                     "wb").write(tflite_model)
            exit()
        self.postProcessModule = save
